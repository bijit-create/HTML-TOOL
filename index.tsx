/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleGenAI, Type, Modality } from '@google/genai';
import { sanitizeHtml } from 'safevalues';
// FIX: 'setElementText' is not an exported member of 'safevalues/dom', so we only import what's used.
import { setElementInnerHtml } from 'safevalues/dom';
import JSZip from 'jszip';
import MarkdownIt from 'markdown-it';
import texmath from 'markdown-it-texmath';
import katex from 'katex';

const md = new MarkdownIt({
    html: true,
    linkify: true,
    typographer: true
}).use(texmath, {
    engine: katex,
    delimiters: 'dollars',
    katexOptions: {
        macros: { "\\RR": "\\mathbb{R}" },
        strict: false
    }
});

const INDIC_CHAR_RANGES = '\\u0900-\\u097F\\u0980-\\u09FF\\u0A80-\\u0AFF\\u0C00-\\u0C7F';
const INDIC_TEXT_REGEX = new RegExp(`[${INDIC_CHAR_RANGES}]`);

function wrapIndicTextInLatex(mathContent: string): string {
    if (!INDIC_TEXT_REGEX.test(mathContent)) return mathContent;

    const protectedTextCommands: string[] = [];
    let normalized = mathContent.replace(/\\(?:text|mbox|mathrm)\{[^{}]*\}/g, (match) => {
        const index = protectedTextCommands.push(match) - 1;
        return `@@TEXT_CMD_${index}@@`;
    });

    const indicInBraceGroup = new RegExp(`\\{([^{}]*[${INDIC_CHAR_RANGES}][^{}]*)\\}`, 'g');
    normalized = normalized.replace(indicInBraceGroup, (match, group: string) => {
        if (group.includes('\\')) return match;
        const trimmed = group.trim();
        return trimmed ? `{\\text{${trimmed}}}` : match;
    });

    const indicTextNearOperator = new RegExp(`(^|[=+\\-*/^_(),:;])\\s*([${INDIC_CHAR_RANGES}][${INDIC_CHAR_RANGES}\\s\\u200c\\u200d\\u0964.,'"!?-]*[${INDIC_CHAR_RANGES}])(?=\\s*($|[=+\\-*/^_(),:;]))`, 'g');
    normalized = normalized.replace(indicTextNearOperator, (match, prefix: string, phrase: string) => {
        const trimmed = phrase.trim();
        return trimmed ? `${prefix}\\text{${trimmed}}` : match;
    });

    return normalized.replace(/@@TEXT_CMD_(\d+)@@/g, (_, index) => protectedTextCommands[Number(index)] ?? '');
}

function normalizeMultilingualLatex(text: string): string {
    if (!text || !INDIC_TEXT_REGEX.test(text) || !text.includes('$')) return text;
    return text.replace(/(\$\$?)([\s\S]*?)\1/g, (match, delimiter: string, mathContent: string) => {
        return `${delimiter}${wrapIndicTextInLatex(mathContent)}${delimiter}`;
    });
}

// --- DATA STRUCTURES ---

interface BilingualText {
  en: string;
  targetLang: string;
}

interface UiTranslations {
  lessonSummary: string | BilingualText;
  keyTakeaways: string | BilingualText;
  items: string | BilingualText;
  targets: string | BilingualText;
  true: string | BilingualText;
  false: string | BilingualText;
  finishLesson: string | BilingualText;
  letsExplore: string | BilingualText;
  back: string | BilingualText;
  correctFeedback: string | BilingualText;
  incorrectFeedback: string | BilingualText;
  summary: string | BilingualText;
  page: string | BilingualText;
  done: string | BilingualText;
  completionTitle?: string | BilingualText;
  completionMessage?: string | BilingualText;
}


// --- DOM ELEMENT REFERENCES ---
interface Lesson {
  title: string | BilingualText;
  engagingQuestion: string | BilingualText;
  steps: LessonStep[];
  keyTakeaways: (string | BilingualText)[];
  uiTranslations: UiTranslations;
}

interface LessonStep {
  step: number;
  title: string | BilingualText;
  explanation: string | BilingualText;
  nextStepHint: string | BilingualText;
  image: {
    required: boolean;
    prompt: string;
    // Fields for internal use
    fileName?: string;
    base64Data?: string;
  };
}

// --- DOM ELEMENT REFERENCES ---
const gradeSelect = document.getElementById('grade-select') as HTMLSelectElement;
const subjectSelect = document.getElementById('subject-select') as HTMLSelectElement;
const languageSelect = document.getElementById('language-select') as HTMLSelectElement;
const countrySelect = document.getElementById('country-select') as HTMLSelectElement;
const stateInput = document.getElementById('state-input') as HTMLInputElement;
const pdfUpload = document.getElementById('pdf-upload') as HTMLInputElement;
const videoUpload = document.getElementById('video-upload') as HTMLInputElement;
const videoLink = document.getElementById('video-link') as HTMLInputElement;
const topicInput = document.getElementById('topic-input') as HTMLInputElement;
const specificPromptInput = document.getElementById('specific-prompt-input') as HTMLTextAreaElement;
const pagesInput = document.getElementById('pages-input') as HTMLInputElement;
const pagesValue = document.getElementById('pages-value') as HTMLSpanElement;
const imageUpload = document.getElementById('image-upload') as HTMLInputElement;
const generateBtn = document.getElementById('generate-btn') as HTMLButtonElement;
const loader = document.getElementById('loader') as HTMLDivElement;
const loadingStatus = document.getElementById('loading-status') as HTMLParagraphElement;
const placeholder = document.getElementById('placeholder') as HTMLDivElement;
const lessonPreviewContainer = document.getElementById('lesson-preview-container') as HTMLDivElement;
const lessonPreview = document.getElementById('lesson-preview') as HTMLDivElement;
const downloadBtn = document.getElementById('download-btn') as HTMLButtonElement;
const errorMessage = document.getElementById('error-message') as HTMLDivElement;
const generateBtnText = generateBtn.querySelector('.btn-text') as HTMLSpanElement;

// --- STATE ---
let currentLessonData: Lesson | null = null;
let ai: GoogleGenAI;

// --- AUTO TRANSLATION STATE ---
const AUTO_TRANSLATE_DEBOUNCE_MS = 900;
const autoTranslationTimers = new Map<string, number>();
const autoTranslationVersions = new Map<string, number>();

// --- AUDIO STATE ---
const audioCache = new Map<string, AudioBuffer>();
let currentAudioSource: AudioBufferSourceNode | null = null;
let outputAudioContext: AudioContext;
let currentPlayingButton: HTMLButtonElement | null = null;


// --- Initialize Gemini API ---
try {
  ai = new GoogleGenAI({ apiKey: process.env.API_KEY! });
} catch (error) {
  console.error(error);
  showError('Failed to initialize the AI service. Please check the API key.');
}

// --- AUDIO FUNCTIONS ---

/**
 * Initializes the Web Audio API AudioContext.
 * Must be called as a result of a user gesture (e.g., a click).
 */
function initializeAudioContext() {
    if (!outputAudioContext) {
        // The TTS model returns audio at a 24000 sample rate.
        outputAudioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
    }
}

/**
 * Decodes a base64 string into a Uint8Array of bytes.
 */
function decode(base64: string) {
    const binaryString = atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes;
}

/**
 * Decodes raw PCM audio data into an AudioBuffer for playback.
 */
async function decodeAudioData(
    data: Uint8Array,
    ctx: AudioContext,
    sampleRate: number,
    numChannels: number,
): Promise<AudioBuffer> {
    // The raw data is 16-bit PCM, so we create a view for it.
    const dataInt16 = new Int16Array(data.buffer);
    const frameCount = dataInt16.length / numChannels;
    const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

    for (let channel = 0; channel < numChannels; channel++) {
        const channelData = buffer.getChannelData(channel);
        for (let i = 0; i < frameCount; i++) {
            // Normalize the 16-bit integer to a float between -1.0 and 1.0
            channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
        }
    }
    return buffer;
}


/**
 * Stops any currently playing audio and resets the corresponding UI button.
 */
function stopCurrentAudio() {
    if (currentAudioSource) {
        currentAudioSource.stop();
        currentAudioSource.disconnect();
        currentAudioSource = null;
    }
    if (currentPlayingButton) {
        currentPlayingButton.classList.remove('playing', 'loading');
        currentPlayingButton.innerHTML = `<i class="fa-solid fa-volume-high"></i>`;
        currentPlayingButton.title = 'Read Aloud';
        currentPlayingButton.disabled = false;
        currentPlayingButton = null;
    }
}

/**
 * Plays an AudioBuffer and updates the button's UI state.
 */
function playAudio(buffer: AudioBuffer, button: HTMLButtonElement) {
    stopCurrentAudio(); // Ensure nothing else is playing

    const source = outputAudioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(outputAudioContext.destination);
    source.start(0);

    currentAudioSource = source;
    currentPlayingButton = button;

    button.classList.remove('loading');
    button.classList.add('playing');
    button.innerHTML = `<i class="fa-solid fa-pause"></i>`;
    button.title = 'Stop';
    button.disabled = false;

    // When the audio finishes playing, reset the state.
    source.onended = () => {
        // Only reset if this button is still the one marked as playing.
        if (currentPlayingButton === button) {
            stopCurrentAudio();
        }
    };
}

function stripHtmlToText(text: string): string {
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = text;
    return tempDiv.innerText || text;
}

function getLessonTextForCurrentLanguage(data: string | BilingualText, isBilingual = languageSelect.value !== 'English'): string {
    if (typeof data === 'string') return data;
    return isBilingual ? data.targetLang : data.en;
}

function getAudioTextForScope(scope: string, stepNum?: string): { text: string; cacheKey: string; fileName: string } {
    if (!currentLessonData) {
        return { text: '', cacheKey: '', fileName: '' };
    }

    const isBilingual = languageSelect.value !== 'English';
    const getText = (data: string | BilingualText) => getLessonTextForCurrentLanguage(data, isBilingual);

    if (scope === 'engaging-question') {
        return {
            text: getText(currentLessonData.engagingQuestion),
            cacheKey: 'engaging-question-full-audio',
            fileName: 'engaging-question.wav'
        };
    }

    if (scope === 'step' && stepNum) {
        const step = currentLessonData.steps[parseInt(stepNum, 10) - 1];
        if (!step) {
            return { text: '', cacheKey: '', fileName: '' };
        }

        const combinedText = [
            stripHtmlToText(getText(step.title)),
            stripHtmlToText(getText(step.explanation)),
            stripHtmlToText(getText(step.nextStepHint))
        ].filter(Boolean).join('. ');

        return {
            text: combinedText,
            cacheKey: `step-${stepNum}-full-audio`,
            fileName: `step-${stepNum}.wav`
        };
    }

    if (scope === 'summary') {
        const summaryTitleText = getText(currentLessonData.uiTranslations.lessonSummary);
        const keyTakeawaysTitleText = getText(currentLessonData.uiTranslations.keyTakeaways);
        const takeawaysText = currentLessonData.keyTakeaways.map(getText).join('. ');

        return {
            text: [`${summaryTitleText}. ${keyTakeawaysTitleText}.`, takeawaysText].filter(Boolean).join(' '),
            cacheKey: 'summary-full-audio',
            fileName: 'summary.wav'
        };
    }

    return { text: '', cacheKey: '', fileName: '' };
}

/**
 * Converts math and markup-heavy lesson text into cleaner speech text for TTS.
 */
function preprocessTextForSpeech(rawText: string): string {
    if (!rawText) return '';

    let text = stripHtmlToText(rawText);
    text = text.replace(/\|\|\|/g, '');

    const mathRegex = /\$\$([\s\S]+?)\$\$|\$([\s\S]+?)\$/g;
    text = text.replace(mathRegex, (_match, blockMath, inlineMath) => {
        let math = blockMath || inlineMath || '';

        math = math.replace(/\\frac\{([^}]+)\}\{([^}]+)\}/g, '$1 over $2');
        math = math.replace(/\\sqrt\{([^}]+)\}/g, 'square root of $1');
        math = math.replace(/\+/g, ' plus ');
        math = math.replace(/\s*-\s*/g, ' minus ');
        math = math.replace(/\\times/g, ' times ');
        math = math.replace(/\s*\*\s*/g, ' times ');
        math = math.replace(/\\cdot/g, ' times ');
        math = math.replace(/\\div/g, ' divided by ');
        math = math.replace(/\s*\/\s*/g, ' divided by ');
        math = math.replace(/\s*=\s*/g, ' equals ');
        math = math.replace(/\s*<\s*/g, ' less than ');
        math = math.replace(/\s*>\s*/g, ' greater than ');
        math = math.replace(/\\leq|\\le/g, ' less than or equal to ');
        math = math.replace(/\\geq|\\ge/g, ' greater than or equal to ');
        math = math.replace(/\\neq|\\ne/g, ' not equal to ');
        math = math.replace(/\^2/g, ' squared ');
        math = math.replace(/\^3/g, ' cubed ');
        math = math.replace(/\^(\d+)/g, ' to the power of $1 ');
        math = math.replace(/\^\{([^}]+)\}/g, ' to the power of $1 ');
        math = math.replace(/\\pi/gi, ' pi ');
        math = math.replace(/\\theta/gi, ' theta ');
        math = math.replace(/\\sin/gi, ' sine ');
        math = math.replace(/\\cos/gi, ' cosine ');
        math = math.replace(/\\tan/gi, ' tangent ');
        math = math.replace(/\\log/gi, ' log ');
        math = math.replace(/\\ln/gi, ' natural log ');
        math = math.replace(/\\(?:text|mbox|mathrm)\{([^}]+)\}/g, '$1');
        math = math.replace(/\\[a-zA-Z]+/g, (match) => ` ${match.substring(1)} `);
        math = math.replace(/[\\{}\[\]()_]/g, ' ');

        return math;
    });

    text = text.replace(/\$(\d+(\.\d+)?)/g, '$1 dollars');
    text = text.replace(/\$/g, '');
    return text.replace(/\s+/g, ' ').trim();
}


/**
 * Reusable function to call the Gemini TTS API.
 * @param textToSpeak The raw text to be spoken.
 * @returns A promise that resolves with the base64 encoded audio string, or null on failure.
 */
async function getTtsAudio(rawText: string): Promise<string | null> {
    const processedText = preprocessTextForSpeech(rawText);
    const textToSpeak = `Read the following aloud in a clear, professional female voice with an Indian English accent. Pronounce mathematical symbols and equations naturally. Do not say "dollar sign" for LaTeX delimiters. Follow the natural phrasing of the sentence:
"${processedText}"`;

    const response = await ai.models.generateContent({
        model: "gemini-2.5-flash-preview-tts",
        contents: [{ parts: [{ text: textToSpeak }] }],
        config: {
            responseModalities: [Modality.AUDIO],
            speechConfig: {
                voiceConfig: {
                    // 'Kore' is a clear female voice that responds well to prompting.
                    prebuiltVoiceConfig: { voiceName: 'Kore' },
                },
            },
        },
    });
    return response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data ?? null;
}


/**
 * Handles clicks on any page-level "Read Aloud" button.
 * It generates, caches, and plays audio for all text on that page.
 */
async function handlePageAudioButtonClick(event: Event) {
    initializeAudioContext();
    const button = (event.currentTarget as HTMLButtonElement);

    // If this button's audio is already playing, stop it.
    if (button.classList.contains('playing')) {
        stopCurrentAudio();
        return;
    }

    // Stop any other audio before starting a new one.
    stopCurrentAudio();

    const { text: combinedText, cacheKey } = getAudioTextForScope(button.dataset.scope!, button.dataset.step);

    if (!combinedText || combinedText.trim().length === 0) return;

    // Set UI to loading state
    button.classList.add('loading');
    button.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i>`;
    button.title = 'Loading...';
    button.disabled = true;
    currentPlayingButton = button;

    try {
        // If we have cached audio, play it immediately.
        if (audioCache.has(cacheKey)) {
            playAudio(audioCache.get(cacheKey)!, button);
            return;
        }

        const base64Audio = await getTtsAudio(combinedText);

        if (!base64Audio) {
            throw new Error('No audio data received from the API.');
        }

        // Decode the audio data into a playable buffer
        const audioBuffer = await decodeAudioData(
            decode(base64Audio),
            outputAudioContext,
            24000, // The TTS model returns audio at this sample rate.
            1      // The audio is mono.
        );

        audioCache.set(cacheKey, audioBuffer);
        playAudio(audioBuffer, button);

    } catch (error) {
        console.error("Audio generation failed:", error);
        showError(getFriendlyErrorMessage("Failed to generate audio", error));
        stopCurrentAudio(); // Reset button state on error
    }
}

/**
 * Regenerates audio from the latest edited text and updates the playback cache.
 */
async function handlePageAudioUpdateClick(event: Event) {
    initializeAudioContext();
    const button = event.currentTarget as HTMLButtonElement;

    stopCurrentAudio();

    const { text: combinedText, cacheKey } = getAudioTextForScope(button.dataset.scope!, button.dataset.step);
    if (!combinedText || combinedText.trim().length === 0) return;

    const originalHtml = button.innerHTML;
    button.classList.add('loading');
    button.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i>`;
    button.title = 'Updating Audio...';
    button.disabled = true;

    try {
        const base64Audio = await getTtsAudio(combinedText);
        if (!base64Audio) {
            throw new Error('No audio data received from the API.');
        }

        const audioBuffer = await decodeAudioData(
            decode(base64Audio),
            outputAudioContext,
            24000,
            1
        );

        audioCache.set(cacheKey, audioBuffer);
        button.innerHTML = `<i class="fa-solid fa-check"></i>`;
        button.title = 'Audio updated';

        const parentGroup = button.closest('.audio-controls-group');
        const playBtn = parentGroup?.querySelector('.audio-btn') as HTMLButtonElement | null;
        playAudio(audioBuffer, playBtn || button);

        setTimeout(() => {
            button.classList.remove('loading');
            button.innerHTML = originalHtml;
            button.title = 'Update/Regenerate Audio';
            button.disabled = false;
        }, 1500);
    } catch (error) {
        console.error("Audio update failed:", error);
        showError(getFriendlyErrorMessage("Failed to update audio", error));
        stopCurrentAudio();
        button.classList.remove('loading');
        button.innerHTML = originalHtml;
        button.title = 'Update/Regenerate Audio';
        button.disabled = false;
    }
}


/**
 * Creates a group of audio controls (Play and Download) for a given page or step.
 */
function createPageAudioControls(scope: 'engaging-question' | 'step' | 'summary', stepNum?: number): HTMLDivElement {
    const container = document.createElement('div');
    container.className = 'audio-controls-group';
    
    const playBtn = createPageAudioButton(scope, stepNum);

    const updateBtn = document.createElement('button');
    updateBtn.className = 'audio-update-btn';
    updateBtn.innerHTML = `<i class="fa-solid fa-arrows-rotate"></i>`;
    updateBtn.setAttribute('aria-label', 'Update audio to reflect text changes');
    updateBtn.title = 'Update/Regenerate Audio';
    updateBtn.dataset.scope = scope;
    if (stepNum) {
        updateBtn.dataset.step = String(stepNum);
    }
    updateBtn.addEventListener('click', handlePageAudioUpdateClick);
    
    const downloadBtn = document.createElement('button');
    downloadBtn.className = 'audio-download-btn';
    downloadBtn.innerHTML = `<i class="fa-solid fa-download"></i>`;
    downloadBtn.setAttribute('aria-label', 'Download audio for this section');
    downloadBtn.title = 'Download Audio';
    downloadBtn.dataset.scope = scope;
    if (stepNum) {
        downloadBtn.dataset.step = String(stepNum);
    }
    downloadBtn.addEventListener('click', handleIndividualAudioDownload);
    
    container.appendChild(playBtn);
    container.appendChild(updateBtn);
    container.appendChild(downloadBtn);
    return container;
}

/**
 * Creates a "Read Aloud" button for a given page or step.
 */
function createPageAudioButton(scope: 'engaging-question' | 'step' | 'summary', stepNum?: number): HTMLButtonElement {
    const button = document.createElement('button');
    button.className = 'audio-btn';
    button.innerHTML = `<i class="fa-solid fa-volume-high"></i>`;
    button.setAttribute('aria-label', 'Read page text aloud');
    button.title = 'Read Aloud';
    button.dataset.scope = scope;
    if (stepNum) {
        button.dataset.step = String(stepNum);
    }
    button.addEventListener('click', handlePageAudioButtonClick);
    return button;
}

/**
 * Updates the UI to show download links for audio files that failed to generate.
 * @param failedIds A list of IDs ('engaging-question', 'step-X', 'summary') for failed audio tasks.
 */
function updateUiWithFailedAudioLinks(failedIds: string[]) {
    // First, clear any existing failed links
    document.querySelectorAll('.download-missing-audio-btn').forEach(btn => btn.remove());

    if (failedIds.length === 0) return;

    showError(`Some audio files could not be generated and were excluded from the ZIP file. You can try downloading them individually.`);

    failedIds.forEach(id => {
        let scope: string;
        let stepNum: string | undefined;

        if (id.startsWith('step-')) {
            scope = 'step';
            stepNum = id.split('-')[1];
        } else if (id.startsWith('summary-')) {
            scope = 'summary';
        } else {
            scope = id;
        }

        const parentButton = document.querySelector(`.audio-btn[data-scope="${scope}"]` + (stepNum ? `[data-step="${stepNum}"]` : '')) as HTMLElement;
        
        if (parentButton && parentButton.parentElement) {
            const downloadBtn = document.createElement('button');
            downloadBtn.className = 'download-missing-audio-btn';
            downloadBtn.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Download Missing Audio`;
            downloadBtn.dataset.scope = scope;
            if (stepNum) {
                downloadBtn.dataset.step = stepNum;
            }
            downloadBtn.onclick = handleIndividualAudioDownload;

            // Insert after the button
            parentButton.insertAdjacentElement('afterend', downloadBtn);
        }
    });
}

/**
 * Handles the click event to download a single missing audio file.
 */
async function handleIndividualAudioDownload(event: Event) {
    const button = event.currentTarget as HTMLButtonElement;
    const isIconButton = button.classList.contains('audio-download-btn');
    const originalHtml = button.innerHTML;
    
    button.disabled = true;
    button.innerHTML = isIconButton ? `<i class="fa-solid fa-spinner fa-spin"></i>` : `<i class="fa-solid fa-spinner fa-spin"></i> Generating...`;

    const { text: textToSpeak, fileName } = getAudioTextForScope(button.dataset.scope!, button.dataset.step);

    if (!textToSpeak.trim()) {
        showError('No text content found to generate audio.');
        button.disabled = false;
        button.innerHTML = originalHtml;
        return;
    }

    try {
        const base64Audio = await getTtsAudio(textToSpeak);
        if (!base64Audio) {
            throw new Error('API returned no audio data.');
        }

        const pcmData = decode(base64Audio);
        const wavBlob = createWavFileBlob(pcmData, 24000);

        const url = URL.createObjectURL(wavBlob);
        const a = document.createElement("a");
        a.href = url;
        a.download = fileName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        if (isIconButton) {
            button.disabled = false;
            button.innerHTML = originalHtml;
        } else {
            button.remove();
        }

    } catch (err) {
        console.error(`Individual audio download failed for ${fileName}:`, err);
        showError(getFriendlyErrorMessage(`Could not generate audio for this section (${fileName})`, err));
        button.disabled = false;
        button.innerHTML = originalHtml;
    }
}


// --- CORE FUNCTIONS ---

/**
 * Sets the UI to a loading state.
 */
function setLoading(isLoading: boolean, status: string = '') {
  generateBtn.disabled = isLoading;

  if (isLoading) {
    // When loading starts, hide all other panels and show the loader
    placeholder.style.display = 'none';
    lessonPreviewContainer.style.display = 'none';
    loader.style.display = 'block';
    generateBtnText.textContent = 'Generating...';
    // FIX: Replaced `setElementText` with the standard `textContent` property,
    // as `setElementText` is not a valid export from `safevalues/dom`.
    loadingStatus.textContent = status;
  } else {
    // When loading ends, just hide the loader and reset the button text
    loader.style.display = 'none';
    generateBtnText.textContent = 'Generate Lesson';
  }
}

/**
 * Hides and clears the error message box.
 */
function clearError() {
    errorMessage.style.display = 'none';
    setElementInnerHtml(errorMessage, sanitizeHtml(''));
}

/**
 * Displays an error message in the UI.
 */
function showError(message: string) {
  setElementInnerHtml(errorMessage, sanitizeHtml(message));
  errorMessage.style.display = 'block';
}

function getFriendlyErrorMessage(prefix: string, error: any): string {
    const rawMessage = [
        typeof error === 'string' ? error : '',
        error?.message || '',
        (() => {
            try {
                return JSON.stringify(error);
            } catch {
                return '';
            }
        })()
    ].filter(Boolean).join(' ');

    const lowerMessage = rawMessage.toLowerCase();
    const isQuotaError = rawMessage.includes('429') ||
        rawMessage.includes('RESOURCE_EXHAUSTED') ||
        lowerMessage.includes('quota') ||
        lowerMessage.includes('rate limit');

    if (isQuotaError) {
        return `${prefix}: Gemini API quota or rate limit was reached. Please try again later or check the API key billing/quota settings.`;
    }

    if (lowerMessage.includes('api key') || lowerMessage.includes('permission') || lowerMessage.includes('unauthorized')) {
        return `${prefix}: the AI service could not authenticate. Please check the Gemini API key configuration.`;
    }

    if (lowerMessage.includes('no audio data')) {
        return `${prefix}: the AI service did not return audio data. Please try again.`;
    }

    if (lowerMessage.includes('no image') || lowerMessage.includes('did not return an image')) {
        return `${prefix}: the AI service did not return an image. Try a simpler image prompt.`;
    }

    const compactMessage = rawMessage.replace(/\s+/g, ' ').trim();
    return compactMessage ? `${prefix}: ${compactMessage}` : `${prefix}.`;
}

function updateNestedValue(obj: any, path: string, newValue: any, lang?: keyof BilingualText) {
    const fields = path.split('.');
    let current = obj;
    for (let i = 0; i < fields.length - 1; i++) {
        if (current[fields[i]] === undefined) return;
        current = current[fields[i]];
    }

    const finalField = fields[fields.length - 1];
    if (lang) {
        if (typeof current[finalField] !== 'object' || current[finalField] === null) {
            current[finalField] = { en: '', targetLang: '' };
        }
        current[finalField][lang] = newValue;
        return;
    }

    current[finalField] = newValue;
}

function getAutoTranslationKey(fieldPath: string, stepAttr?: string, indexAttr?: string): string {
    return [fieldPath, stepAttr || 'lesson', indexAttr || 'single'].join(':');
}

function getTargetLanguageEditor(sourceElement: HTMLElement): HTMLElement | null {
    return sourceElement
        .closest('.bilingual-field')
        ?.querySelector('[contenteditable="true"][data-lang="targetLang"]') as HTMLElement | null;
}

function setTargetLanguageText(
    fieldPath: string,
    translatedText: string,
    stepAttr?: string,
    indexAttr?: string
) {
    if (!currentLessonData) return;

    if (fieldPath === 'keyTakeaways' && indexAttr) {
        const index = parseInt(indexAttr, 10);
        const takeaway = currentLessonData.keyTakeaways[index];
        if (typeof takeaway === 'object' && takeaway !== null) {
            takeaway.targetLang = translatedText;
        } else {
            currentLessonData.keyTakeaways[index] = {
                en: typeof takeaway === 'string' ? takeaway : '',
                targetLang: translatedText
            };
        }
        return;
    }

    if (stepAttr) {
        const stepIndex = parseInt(stepAttr, 10) - 1;
        if (!currentLessonData.steps[stepIndex]) return;
        updateNestedValue(currentLessonData.steps[stepIndex], fieldPath, translatedText, 'targetLang');
        return;
    }

    updateNestedValue(currentLessonData, fieldPath, translatedText, 'targetLang');
}

function cancelPendingAutoTranslation(fieldPath: string, stepAttr?: string, indexAttr?: string, targetElement?: HTMLElement) {
    const key = getAutoTranslationKey(fieldPath, stepAttr, indexAttr);
    const timer = autoTranslationTimers.get(key);
    if (timer) {
        window.clearTimeout(timer);
        autoTranslationTimers.delete(key);
    }

    autoTranslationVersions.set(key, (autoTranslationVersions.get(key) || 0) + 1);
    targetElement?.classList.remove('auto-translation-pending', 'auto-translating', 'auto-translation-error');
    targetElement?.removeAttribute('aria-busy');
}

async function translateEditedEnglishText(sourceText: string, targetLanguage: string, preserveHtml: boolean): Promise<string> {
    const formatInstruction = preserveHtml
        ? `The input may contain HTML produced from markdown, KaTeX math markup, key-term spans, and LaTeX delimiters. Translate only the human-readable English lesson text into ${targetLanguage}. Preserve HTML tags, attributes, tag nesting, KaTeX markup, LaTeX math, numbers, variables, URLs, and image/file references exactly.`
        : `Translate the plain English lesson text into ${targetLanguage}. Preserve LaTeX math, numbers, variables, URLs, and punctuation structure where appropriate.`;

    const response = await ai.models.generateContent({
        model: 'gemini-3-flash-preview',
        contents: {
            parts: [{
                text: `You are updating one edited lesson field. ${formatInstruction}
Return JSON only with this shape: {"translatedText":"..."}.
Do not add explanations, labels, markdown fences, or extra keys.

English source:
${sourceText}`
            }]
        },
        config: {
            responseMimeType: 'application/json',
            responseSchema: {
                type: Type.OBJECT,
                properties: {
                    translatedText: { type: Type.STRING }
                },
                required: ['translatedText']
            }
        }
    });

    const parsed = JSON.parse(response.text || '{}');
    const translatedText = cleanDataObject(parsed.translatedText || '');
    if (typeof translatedText !== 'string') {
        throw new Error('The AI returned an invalid translation response.');
    }

    return normalizeMultilingualLatex(translatedText);
}

function applyAutoTranslationResult(
    targetElement: HTMLElement | null,
    translatedText: string,
    preserveHtml: boolean,
    fieldPath: string,
    stepAttr?: string,
    indexAttr?: string
) {
    setTargetLanguageText(fieldPath, translatedText, stepAttr, indexAttr);

    if (targetElement?.isConnected) {
        if (preserveHtml) {
            targetElement.innerHTML = translatedText;
        } else {
            targetElement.textContent = translatedText;
        }
        targetElement.classList.remove('auto-translation-pending', 'auto-translating', 'auto-translation-error');
        targetElement.removeAttribute('aria-busy');
    }

    saveAppState();
}

function scheduleAutoTranslationFromEnglishEdit(
    sourceElement: HTMLElement,
    fieldPath: string,
    stepAttr?: string,
    indexAttr?: string
) {
    if (!ai || languageSelect.value === 'English') return;

    const key = getAutoTranslationKey(fieldPath, stepAttr, indexAttr);
    const preserveHtml = fieldPath !== 'keyTakeaways';
    const sourceText = preserveHtml ? sourceElement.innerHTML : sourceElement.innerText;
    const targetElement = getTargetLanguageEditor(sourceElement);
    const version = (autoTranslationVersions.get(key) || 0) + 1;

    const existingTimer = autoTranslationTimers.get(key);
    if (existingTimer) {
        window.clearTimeout(existingTimer);
    }

    autoTranslationVersions.set(key, version);
    targetElement?.classList.remove('auto-translating', 'auto-translation-error');
    targetElement?.classList.add('auto-translation-pending');

    if (!stripHtmlToText(sourceText).trim()) {
        autoTranslationTimers.delete(key);
        applyAutoTranslationResult(targetElement, '', preserveHtml, fieldPath, stepAttr, indexAttr);
        return;
    }

    const timer = window.setTimeout(async () => {
        autoTranslationTimers.delete(key);
        targetElement?.classList.remove('auto-translation-pending');
        targetElement?.classList.add('auto-translating');
        targetElement?.setAttribute('aria-busy', 'true');

        try {
            const translatedText = await translateEditedEnglishText(sourceText, languageSelect.value, preserveHtml);
            if (autoTranslationVersions.get(key) !== version) return;

            applyAutoTranslationResult(targetElement, translatedText, preserveHtml, fieldPath, stepAttr, indexAttr);
        } catch (error) {
            if (autoTranslationVersions.get(key) !== version) return;

            console.error('Auto translation failed:', error);
            targetElement?.classList.remove('auto-translation-pending', 'auto-translating');
            targetElement?.classList.add('auto-translation-error');
            targetElement?.removeAttribute('aria-busy');
            showError(getFriendlyErrorMessage('Translation update failed', error));
        }
    }, AUTO_TRANSLATE_DEBOUNCE_MS);

    autoTranslationTimers.set(key, timer);
}

/**
 * Reads an uploaded file and returns its Base64 representation and mime type.
 */
function fileToBase64(file: File): Promise<{base64: string, mimeType: string}> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
        const result = reader.result as string;
        const parts = result.split(',');
        const mimeType = parts[0].match(/:(.*?);/)![1];
        const base64 = parts[1];
        resolve({ base64, mimeType });
    };
    reader.onerror = (error) => reject(error);
  });
}

/**
 * Compresses an image by resizing and converting to JPEG.
 * @param base64Data The base64 string of the image (without the data URL prefix).
 * @param mimeType The original mime type of the image.
 * @returns A promise that resolves with the compressed base64 string (JPEG).
 */
async function compressImage(base64Data: string, mimeType: string, quality = 0.8): Promise<string> {
    const dataUrl = `data:${mimeType};base64,${base64Data}`;
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            const MAX_WIDTH = 1280; // Resize for web optimization and size reduction
            let width = img.width;
            let height = img.height;

            if (width > MAX_WIDTH) {
                const scale = MAX_WIDTH / width;
                width = MAX_WIDTH;
                height = height * scale;
            }

            canvas.width = width;
            canvas.height = height;
            const ctx = canvas.getContext('2d');
            if (!ctx) {
                return reject(new Error('Could not get canvas context'));
            }
            ctx.drawImage(img, 0, 0, width, height);
            
            // Convert to JPEG for smaller file size, which is better for web.
            const compressedDataUrl = canvas.toDataURL('image/jpeg', quality);
            resolve(compressedDataUrl.split(',')[1]); // Return only base64 part
        };
        img.onerror = reject;
        img.src = dataUrl;
    });
}

/**
 * Recursively traverses an object or array and cleans all string values
 * by removing non-breaking space entities and trimming whitespace.
 * @param data The object or array to clean.
 * @returns A new object or array with cleaned string values.
 */
function cleanDataObject(data: any): any {
    if (typeof data === 'string') {
        return data.replace(/&nbsp;/g, ' ').trim();
    }
    if (Array.isArray(data)) {
        return data.map(cleanDataObject);
    }
    if (typeof data === 'object' && data !== null) {
        const newData: { [key: string]: any } = {};
        for (const key in data) {
            if (Object.prototype.hasOwnProperty.call(data, key)) {
                newData[key] = cleanDataObject(data[key]);
            }
        }
        return newData;
    }
    return data;
}

function normalizeLessonMathSyntax<T>(data: T): T {
    if (typeof data === 'string') {
        return normalizeMultilingualLatex(data) as T;
    }
    if (Array.isArray(data)) {
        return data.map(normalizeLessonMathSyntax) as T;
    }
    if (typeof data === 'object' && data !== null) {
        const normalizedData: { [key: string]: any } = {};
        for (const key in data) {
            if (Object.prototype.hasOwnProperty.call(data, key)) {
                normalizedData[key] = normalizeLessonMathSyntax((data as any)[key]);
            }
        }
        return normalizedData as T;
    }
    return data;
}

function getTextValue(data: any): string {
    if (typeof data === 'string') return data;
    if (data && typeof data === 'object') {
        return data.targetLang || data.en || '';
    }
    return '';
}

function toSafeAsciiFileName(input: string, fallback = 'lesson'): string {
    const safeName = input
        .normalize('NFKD')
        .replace(/[^\x00-\x7F]/g, '')
        .replace(/[^A-Za-z0-9._-]+/g, '_')
        .replace(/_+/g, '_')
        .replace(/^[_ .-]+|[_ .-]+$/g, '')
        .slice(0, 80);

    return safeName || fallback;
}

function getHtmlLangCode(language: string): string {
    const langMap: Record<string, string> = {
        English: 'en',
        Hindi: 'hi',
        Marathi: 'mr',
        Gujarati: 'gu',
        Telugu: 'te'
    };

    return langMap[language] || 'en';
}

const APP_STATE_KEYS = {
    lessonData: 'aistudio_lesson_generator_data',
    topic: 'aistudio_lesson_generator_topic',
    grade: 'aistudio_lesson_generator_grade',
    subject: 'aistudio_lesson_generator_subject',
    language: 'aistudio_lesson_generator_language',
    country: 'aistudio_lesson_generator_country',
    state: 'aistudio_lesson_generator_state',
    specificPrompt: 'aistudio_lesson_generator_specific_prompt',
    pages: 'aistudio_lesson_generator_pages',
    videoLink: 'aistudio_lesson_generator_video_link'
};

function saveAppState() {
    try {
        if (currentLessonData) {
            localStorage.setItem(APP_STATE_KEYS.lessonData, JSON.stringify(currentLessonData));
        } else {
            localStorage.removeItem(APP_STATE_KEYS.lessonData);
        }

        localStorage.setItem(APP_STATE_KEYS.topic, topicInput.value || '');
        localStorage.setItem(APP_STATE_KEYS.grade, gradeSelect.value || '');
        localStorage.setItem(APP_STATE_KEYS.subject, subjectSelect.value || '');
        localStorage.setItem(APP_STATE_KEYS.language, languageSelect.value || '');
        localStorage.setItem(APP_STATE_KEYS.country, countrySelect.value || '');
        localStorage.setItem(APP_STATE_KEYS.state, stateInput.value || '');
        localStorage.setItem(APP_STATE_KEYS.specificPrompt, specificPromptInput.value || '');
        localStorage.setItem(APP_STATE_KEYS.pages, pagesInput.value || '');
        localStorage.setItem(APP_STATE_KEYS.videoLink, videoLink.value || '');
    } catch (error) {
        console.error('Failed to save state to localStorage:', error);
    }
}

function restoreInputValue(element: HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement, value: string | null) {
    if (value !== null) {
        element.value = value;
    }
}

function loadAppState() {
    try {
        restoreInputValue(topicInput, localStorage.getItem(APP_STATE_KEYS.topic));
        restoreInputValue(gradeSelect, localStorage.getItem(APP_STATE_KEYS.grade));
        restoreInputValue(subjectSelect, localStorage.getItem(APP_STATE_KEYS.subject));
        restoreInputValue(languageSelect, localStorage.getItem(APP_STATE_KEYS.language));
        restoreInputValue(countrySelect, localStorage.getItem(APP_STATE_KEYS.country));
        restoreInputValue(stateInput, localStorage.getItem(APP_STATE_KEYS.state));
        restoreInputValue(specificPromptInput, localStorage.getItem(APP_STATE_KEYS.specificPrompt));
        restoreInputValue(pagesInput, localStorage.getItem(APP_STATE_KEYS.pages));
        restoreInputValue(videoLink, localStorage.getItem(APP_STATE_KEYS.videoLink));
        pagesValue.textContent = pagesInput.value;

        const savedLessonData = localStorage.getItem(APP_STATE_KEYS.lessonData);
        if (savedLessonData) {
            currentLessonData = normalizeLessonMathSyntax(cleanDataObject(JSON.parse(savedLessonData)));
            renderPreview(currentLessonData!);
            placeholder.style.display = 'none';
            lessonPreviewContainer.style.display = 'flex';
        }
    } catch (error) {
        console.error('Failed to load state from localStorage:', error);
        localStorage.removeItem(APP_STATE_KEYS.lessonData);
    }
}


/**
 * The main function to generate the lesson plan.
 */
async function generateLesson() {
  if (!ai) {
    showError('AI service is not available.');
    return;
  }

  const grade = gradeSelect.value;
  const subject = subjectSelect.value;
  const language = languageSelect.value;
  const country = countrySelect.value;
  const state = stateInput.value.trim();
  const topic = topicInput.value.trim();
  const specificPrompt = specificPromptInput.value.trim();
  const pages = pagesInput.value;
  const uploadedFiles = imageUpload.files ?? [];
  const uploadedPdf = pdfUpload.files?.[0];
  const uploadedVideo = videoUpload.files?.[0];
  const videoUrl = videoLink.value.trim();

  if (uploadedVideo && uploadedVideo.size > 100 * 1024 * 1024) {
    showError('Video file size exceeds the 100MB limit. Please provide a smaller file or a video link.');
    return;
  }

  if (!topic && !uploadedVideo && !videoUrl) {
    showError('Please enter a topic or provide a video for context.');
    return;
  }

  clearError();
  // Clear any previous audio state
  stopCurrentAudio();
  audioCache.clear();
  updateUiWithFailedAudioLinks([]); // Clear any old download links
  setLoading(true, 'Generating lesson content...');

  try {
    const isBilingual = language !== 'English';

    const bilingualTextSchema = {
        type: Type.OBJECT,
        properties: {
            en: { type: Type.STRING, description: "The text content in English." },
            targetLang: { type: Type.STRING, description: `The text content in ${language}.` }
        },
        required: ["en", "targetLang"]
    };

    const textSchema = isBilingual ? bilingualTextSchema : { type: Type.STRING };
    const textSchemaWithDesc = (description: string) => isBilingual 
        ? { ...bilingualTextSchema, description } 
        : { type: Type.STRING, description };

    const uiTranslationsSchema = {
        type: Type.OBJECT,
        properties: {
            lessonSummary: textSchema,
            keyTakeaways: textSchema,
            true: textSchema,
            false: textSchema,
            finishLesson: textSchema,
            letsExplore: textSchema,
            back: textSchema,
            correctFeedback: textSchema,
            incorrectFeedback: textSchema,
            summary: textSchema,
            page: textSchema,
            done: textSchema,
            next: textSchema,
            completionTitle: textSchemaWithDesc("Short completion screen heading, translated naturally. Example: Congratulations!"),
            completionMessage: textSchemaWithDesc("Completion screen message before the lesson title, translated naturally. Example: You have completed the lesson:")
        },
        required: ["lessonSummary", "keyTakeaways", "finishLesson", "letsExplore", "back", "completionTitle", "completionMessage"]
    };

    const lessonSchema = {
        type: Type.OBJECT,
        properties: {
            title: textSchema,
            engagingQuestion: textSchemaWithDesc("A thought-provoking question to capture student interest."),
            steps: {
                type: Type.ARRAY,
                items: {
                    type: Type.OBJECT,
                    properties: {
                        step: { type: Type.INTEGER },
                        title: textSchema,
                        explanation: textSchemaWithDesc("Content for the step (under 100 words). Wrap key terms in |||this|||."),
                        nextStepHint: textSchemaWithDesc("A short sentence to bridge to the next step. Empty for the final step."),
                        image: {
                            type: Type.OBJECT,
                            properties: {
                                required: { type: Type.BOOLEAN },
                                prompt: { type: Type.STRING, description: "Detailed prompt for an educational illustration. No text in the image." }
                            },
                            required: ["required", "prompt"],
                        }
                    },
                    required: ["step", "title", "explanation", "nextStepHint", "image"],
                }
            },
            keyTakeaways: {
                type: Type.ARRAY,
                description: "A list of all the key takeaways, one for each step.",
                items: textSchema
            },
            uiTranslations: uiTranslationsSchema,
        },
        required: ["title", "engagingQuestion", "steps", "keyTakeaways", "uiTranslations"],
    };
    
    const contentParts: ({ text: string } | { inlineData: { data: string; mimeType: string } })[] = [];
    let pdfInstruction = '';
    let videoInstruction = '';

    if (uploadedPdf) {
        setLoading(true, 'Reading PDF...');
        const { base64, mimeType } = await fileToBase64(uploadedPdf);
        if (mimeType !== 'application/pdf') {
            showError('Please upload a valid PDF file.');
            setLoading(false);
            return;
        }
        contentParts.push({
            inlineData: {
                data: base64,
                mimeType: mimeType
            }
        });
        pdfInstruction = `A PDF document has been provided as context. You MUST prioritize the information, terminology, and examples from this document when generating the lesson content.`;
        setLoading(true, 'Generating lesson content...'); // Reset status message
    }

    if (uploadedVideo) {
        setLoading(true, `Reading video file (${(uploadedVideo.size / (1024 * 1024)).toFixed(1)} MB)...`);
        const { base64, mimeType } = await fileToBase64(uploadedVideo);
        contentParts.push({
            inlineData: {
                data: base64,
                mimeType: mimeType
            }
        });
        videoInstruction = `A video file has been provided. Analyze the video's script, visual illustrations, and teaching methodology. Extract the core educational content and structure the lesson pages to follow the order and flow of the video. The generated image prompts for each step should be similar in style and content to the illustrations shown in the video.`;
        setLoading(true, 'Generating lesson content...');
    } else if (videoUrl) {
        videoInstruction = `A video link has been provided for context: ${videoUrl}. Please use any available knowledge or context from this video to inform the lesson's structure, script, and illustrations. Match the teaching flow and style found in the video.`;
    }

    let localizationInstruction = `The lesson should be culturally and regionally relevant for students in ${country}.`;
    if (state) {
        localizationInstruction = `The lesson should be culturally and regionally relevant for students in ${state}, ${country}.`;
    }

    const specificInstruction = specificPrompt 
      ? `Give special consideration to this instruction: "${specificPrompt}".`
      : '';

    let languageInstruction = `The entire lesson, including all text, must be in the ${language} language. For example, if the language is Hindi and the context is Indian, use expressions like 'Chaliye, hum vishay ko samjhte hain'.`;
    if (isBilingual) {
        languageInstruction = `The entire lesson needs to be bilingual. For every text field (like title, explanation, question, etc.), you MUST provide a JSON object with two keys: 'en' for the English version, and 'targetLang' for the ${language} version. The 'targetLang' value must be a high-quality, natural-sounding translation. For example, if the language is Hindi and the context is Indian, use expressions like 'Chaliye, hum vishay ko samjhte hain' in the 'targetLang' field.`;
    }

    const textPrompt = `You are an expert curriculum designer specializing in engaging, localized content. Create an impressive, ${pages}-step lesson plan ${topic ? `about "${topic}"` : 'based on the provided video context'} for a grade ${grade} ${subject} class.
    ${localizationInstruction} ${pdfInstruction} ${videoInstruction} Do not mention the country or state name in the lesson content itself; it is only for contextual reference.
    ${specificInstruction}
    ${languageInstruction}
    CRITICAL LATEX GUIDELINES:
    1. For ALL mathematical formulas, variables, or expressions, you MUST use LaTeX notation wrapped in single dollar signs ($...$) for inline math (e.g., $E=mc^2$) and double dollar signs ($$...$$) for block math. This applies to explanations, titles, and takeaways.
    2. For any descriptive words or units inside math environments, you MUST use the \\text{} command (e.g., $l = 22 \\text{ cm}$) so they are rendered as text, not variables.
    3. JSON ESCAPING: In your JSON response, every backslash for LaTeX MUST be double-escaped (e.g., use "\\\\frac" for \frac, "\\\\text" for \text, and "\\\\theta" for \theta). This is vital to prevent character corruption.
    Provide the output in a structured JSON format.
    The lesson must begin with an overall 'engagingQuestion' to spark student curiosity.
    For each step, provide:
    1. A clear title.
    2. A short, concise explanation in markdown format (under 50 words). This is for a 16:9 slide, so brevity is key. In the explanation, identify and wrap all key vocabulary or important concepts in triple pipe delimiters (e.g., |||photosynthesis|||).
    3. A prompt for a suitable educational image that is also culturally relevant to the specified region and inspired by the visual style of the provided video (if applicable).
    4. A short 'nextStepHint' to smoothly transition to the next topic (under 15 words). This should be an empty string for the very last step.
    Finally, at the end of the JSON, provide a 'keyTakeaways' array, containing one short key takeaway sentence for each lesson step.
    Also, provide a 'uiTranslations' object containing translations for standard UI text, including a completionTitle and completionMessage for the final completion screen.`;
      
    contentParts.unshift({ text: textPrompt });

    const response = await ai.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: { parts: contentParts },
      config: {
        responseMimeType: 'application/json',
        responseSchema: lessonSchema
      },
    });

    // FIX: Clean the entire lesson object from the API to remove unwanted entities like &nbsp;
    // This ensures both the preview and the downloaded file have clean data.
    currentLessonData = normalizeLessonMathSyntax(cleanDataObject(JSON.parse(response.text)));

    // FIX: Check for empty or invalid lesson data to prevent blank screen
    if (!currentLessonData || !currentLessonData.steps || currentLessonData.steps.length === 0) {
        throw new Error("The AI returned an empty or invalid lesson plan. Please try refining your topic or generating again.");
    }

    // 2. Process and pre-assign uploaded images
    setLoading(true, 'Processing uploaded images...');
    const uploadedImagesBase64 = await Promise.all(
        Array.from(uploadedFiles).map(async (file) => {
            const { base64, mimeType } = await fileToBase64(file);
            return compressImage(base64, mimeType);
        })
    );

    let userImageIndex = 0;
    const stepsNeedingImages = currentLessonData!.steps.filter(s => s.image.required);
    
    // First, use user-uploaded images sequentially for designated steps
    for (const step of stepsNeedingImages) {
        if (userImageIndex < uploadedImagesBase64.length) {
            step.image.base64Data = uploadedImagesBase64[userImageIndex];
            step.image.fileName = `step_${step.step}.jpg`;
            userImageIndex++;
        }
    }

    // Identify steps that still need AI generated images
    const stepsNeedingAiImages = stepsNeedingImages.filter(step => !step.image.base64Data);
    let completedImages = 0;
    let imageGenerationRateLimited = false;

    if (stepsNeedingAiImages.length > 0) {
        let completedImagesCount = 0;
        let imageGenerationRateLimited = false;

        for (const step of stepsNeedingAiImages) {
            if (imageGenerationRateLimited) break;

            setLoading(true, `Generating AI image ${completedImagesCount + 1} of ${stepsNeedingAiImages.length}...`);

            try {
                const imageResponse = await ai.models.generateContent({
                    model: 'gemini-2.5-flash-image',
                    contents: {
                        parts: [{ text: step.image.prompt }],
                    },
                    config: {
                        responseModalities: [Modality.IMAGE],
                    },
                });

                let rawBase64 = '';
                if (imageResponse.candidates?.[0]?.content?.parts) {
                    for (const part of imageResponse.candidates[0].content.parts) {
                        if (part.inlineData) {
                            rawBase64 = part.inlineData.data;
                            break;
                        }
                    }
                }

                if (rawBase64) {
                    step.image.base64Data = await compressImage(rawBase64, 'image/png');
                    completedImagesCount++;
                }
            } catch (imgError) {
                console.error(`Image generation failed for step ${step.step}:`, imgError);
                const friendlyMessage = getFriendlyErrorMessage(`Image generation failed for step ${step.step}`, imgError);
                if (friendlyMessage.toLowerCase().includes('quota') || friendlyMessage.toLowerCase().includes('rate limit')) {
                    imageGenerationRateLimited = true;
                    showError(friendlyMessage);
                }
            }
            
            step.image.fileName = `step_${step.step}.jpg`;

            // Add a small delay between requests to avoid hitting rate limits
            if (!imageGenerationRateLimited && completedImagesCount < stepsNeedingAiImages.length) {
                await new Promise(resolve => setTimeout(resolve, 2000));
            }
        }
    }

    // 3. Render the live preview
    setLoading(true, 'Finishing up...');
    renderPreview(currentLessonData!);
    lessonPreviewContainer.style.display = 'flex';
    saveAppState();

  } catch (error) {
    console.error('Error generating lesson:', error);
    showError(getFriendlyErrorMessage('An error occurred during lesson generation', error));
    // Ensure other panels are hidden on error
    placeholder.style.display = 'none';
    lessonPreviewContainer.style.display = 'none';
  } finally {
    setLoading(false);
  }
}

function renderBilingualField(
    data: string | BilingualText,
    isBilingual: boolean,
    attributes: Record<string, string>,
    isRichText = false
): string {
    if (isBilingual && typeof data === 'object') {
        const attrString = Object.entries(attributes).map(([key, value]) => `${key}="${value}"`).join(' ');
        const enText = (data.en || '').toString();
        const targetLangText = (data.targetLang || '').toString();

        return `
            <div class="bilingual-field">
                <div class="bilingual-wrapper">
                    <span class="lang-label">EN</span>
                    <div contenteditable="true" ${attrString} data-lang="en">${enText}</div>
                </div>
                <div class="bilingual-wrapper">
                    <span class="lang-label">${languageSelect.value.slice(0, 2).toUpperCase()}</span>
                    <div contenteditable="true" ${attrString} data-lang="targetLang">${targetLangText}</div>
                </div>
            </div>
        `;
    } else {
        const attrString = Object.entries(attributes).map(([key, value]) => `${key}="${value}"`).join(' ');
        const text = (typeof data === 'object' ? (data as BilingualText).en : data) || ''; // Fallback for safety
        const content = isRichText ? text : sanitizeHtml(text);
        return `<div contenteditable="true" ${attrString}>${content}</div>`;
    }
}


/**
 * Renders the lesson data into an editable preview.
 */
function renderPreview(lesson: Lesson) {
  lessonPreview.innerHTML = ''; // Clear previous preview
  const isBilingual = languageSelect.value !== 'English';

  const getUiText = (data: string | BilingualText): string => {
      if (typeof data === 'string') return data;
      return isBilingual ? data.targetLang : data.en;
  };

  const renderContent = (data: string | BilingualText): string | BilingualText => {
      const process = (text: string) => {
          if (!text) return '';
          return md.render(normalizeMultilingualLatex(text)).trim();
      };
      if (typeof data === 'object') {
          return { en: process(data.en), targetLang: process(data.targetLang) };
      }
      return process(data);
  };

  const lessonTitleEl = document.createElement('h1');
  lessonTitleEl.className = 'preview-lesson-title';
  lessonTitleEl.innerHTML = renderBilingualField(renderContent(lesson.title), isBilingual, { 'data-field': 'title' }, true);
  lessonPreview.appendChild(lessonTitleEl);

  const engagingQuestionEl = document.createElement('div');
  engagingQuestionEl.className = 'preview-engaging-question';
  const engagingQuestionContent = renderBilingualField(renderContent(lesson.engagingQuestion), isBilingual, { 'data-field': 'engagingQuestion' }, true);
  engagingQuestionEl.innerHTML = `
      <i class="fa-solid fa-lightbulb"></i>
      <div class="bilingual-container">${engagingQuestionContent}</div>
  `;
  engagingQuestionEl.appendChild(createPageAudioControls('engaging-question'));
  lessonPreview.appendChild(engagingQuestionEl);

  lesson.steps.forEach((step, index) => {
    const stepEl = document.createElement('div');
    stepEl.className = 'preview-step';

    const getExplanationHtml = (explanation: string | BilingualText): string | BilingualText => {
        const process = (text: string) => {
            if (!text) return '';
            // First render with markdown-it which handles LaTeX via the plugin
            let rendered = md.render(normalizeMultilingualLatex(text));
            // Then apply the custom key-term highlighting
            rendered = rendered.replace(/\|\|\|(.*?)\|\|\|/g, '<strong class="key-term">$1</strong>');
            return rendered;
        };
        if (typeof explanation === 'object') {
            return { en: process(explanation.en), targetLang: process(explanation.targetLang) };
        }
        return process(explanation);
    };
    
    const explanationContent = renderBilingualField(getExplanationHtml(step.explanation), isBilingual, { 'data-step': String(step.step), 'data-field': 'explanation' }, true);
    const titleContent = renderBilingualField(step.title, isBilingual, { 'data-step': String(step.step), 'data-field': 'title' });
    
    // FIX: Check for empty hint content before rendering the hint container.
    // The AI might return an object with empty strings for bilingual hints on the last step,
    // which is truthy but should not be rendered.
    const isHintNotEmpty = (hint: string | BilingualText): boolean => {
      if (!hint) return false;
      if (typeof hint === 'string') return hint.trim() !== '';
      // Check if either language has content
      return (hint.en && hint.en.trim() !== '') || (hint.targetLang && hint.targetLang.trim() !== '');
    };
    
    const nextStepHintContent = isHintNotEmpty(step.nextStepHint)
        ? `<div class="preview-next-hint">
               <i class="fa-solid fa-arrow-right-long"></i>
               ${renderBilingualField(step.nextStepHint, isBilingual, { 'data-step': String(step.step), 'data-field': 'nextStepHint' })}
           </div>` 
        : '';

    const stepHeader = document.createElement('div');
    stepHeader.className = 'step-header';
    const stepTitleContainer = document.createElement('h2');
    stepTitleContainer.innerHTML = titleContent;
    stepHeader.appendChild(stepTitleContainer);
    stepHeader.appendChild(createPageAudioControls('step', step.step));
    
    stepEl.innerHTML = `
      <div class="step-controls">
        <button class="move-step-btn" data-index="${index}" data-direction="up" title="Move Up" ${index === 0 ? 'disabled' : ''}><i class="fa-solid fa-arrow-up"></i></button>
        <button class="move-step-btn" data-index="${index}" data-direction="down" title="Move Down" ${index === lesson.steps.length - 1 ? 'disabled' : ''}><i class="fa-solid fa-arrow-down"></i></button>
        <button class="delete-step-btn" data-index="${index}" title="Delete Page"><i class="fa-solid fa-trash-can"></i></button>
      </div>
      <div class="preview-step-image">
        <div id="image-container-${step.step}"></div>
        <div class="image-controls">
            <div class="buttons-row">
                <button class="replace-image-btn" data-step="${step.step}"><i class="fa-solid fa-image"></i> Replace</button>
                <button class="edit-image-btn" data-step="${step.step}" ${!step.image.base64Data ? 'disabled' : ''}><i class="fa-solid fa-wand-magic-sparkles"></i> Edit with AI</button>
            </div>
            <div class="image-edit-controls" id="image-edit-controls-${step.step}" style="display: none;">
                <input type="text" class="image-edit-prompt" placeholder="e.g., Add a retro filter">
                <button class="submit-edit-btn" data-step="${step.step}">Generate</button>
                <button class="cancel-edit-btn" data-step="${step.step}">Cancel</button>
            </div>
            <div class="image-prompt-controls">
                <label for="prompt-input-${step.step}" class="prompt-label">Image Generation Prompt</label>
                <div contenteditable="true" id="prompt-input-${step.step}" class="prompt-input-editable" data-step="${step.step}" data-field="image.prompt">${step.image.prompt}</div>
                <button class="regenerate-image-btn" data-step="${step.step}"><i class="fa-solid fa-arrows-rotate"></i> Regenerate</button>
            </div>
        </div>
      </div>
      <div class="preview-step-content">
        <!-- Step Header will be prepended here -->
        <div class="explanation-container">${explanationContent}</div>
        ${nextStepHintContent}
      </div>
    `;
    stepEl.querySelector('.preview-step-content')!.prepend(stepHeader);
    lessonPreview.appendChild(stepEl);
    
    const imageContainer = stepEl.querySelector(`#image-container-${step.step}`)!;
    renderPreviewImage(imageContainer, step);
    
    // Add event listeners for non-delegated controls
    stepEl.querySelectorAll('[contenteditable="true"]').forEach(el => el.addEventListener('input', handleTextEdit));
    stepEl.querySelector(`button.replace-image-btn`)?.addEventListener('click', handleImageReplace);
    stepEl.querySelector(`.edit-image-btn`)?.addEventListener('click', toggleImageEditControls);
    stepEl.querySelector(`.cancel-edit-btn`)?.addEventListener('click', toggleImageEditControls);
    stepEl.querySelector(`.submit-edit-btn`)?.addEventListener('click', handleImageEdit);
    stepEl.querySelector(`.regenerate-image-btn`)?.addEventListener('click', handleImageRegenerate);
  });

  const summaryEl = document.createElement('div');
  summaryEl.className = 'preview-summary';
  
  const summaryHeader = document.createElement('div');
  summaryHeader.className = 'preview-summary-header';
  const summaryTitle = document.createElement('h3');
  
  if (lesson.uiTranslations) {
      const summaryText = getUiText(lesson.uiTranslations.lessonSummary);
      const takeawaysText = getUiText(lesson.uiTranslations.keyTakeaways);
      summaryTitle.textContent = `${summaryText} (${takeawaysText})`;
  } else {
      summaryTitle.textContent = 'Lesson Summary (Key Takeaways)';
  }

  summaryHeader.appendChild(summaryTitle);
  summaryHeader.appendChild(createPageAudioControls('summary'));
  summaryEl.appendChild(summaryHeader);

  const takeawaysList = document.createElement('ul');
  lesson.keyTakeaways.forEach((takeaway, index) => {
      const li = document.createElement('li');
      li.className = 'takeaway-item';

      const takeawayContent = document.createElement('div');
      takeawayContent.className = 'takeaway-content';
      takeawayContent.innerHTML = renderBilingualField(renderContent(takeaway), isBilingual, { 'data-field': 'keyTakeaways', 'data-index': String(index) }, true);

      const deleteBtn = document.createElement('button');
      deleteBtn.className = 'delete-takeaway-btn';
      deleteBtn.innerHTML = `<i class="fa-solid fa-trash-can"></i>`;
      deleteBtn.title = 'Delete Takeaway';
      deleteBtn.dataset.index = String(index);
      
      li.appendChild(takeawayContent);
      li.appendChild(deleteBtn);

      takeawaysList.appendChild(li);
  });
  summaryEl.appendChild(takeawaysList);
  lessonPreview.appendChild(summaryEl);

  lessonPreview.querySelectorAll('[contenteditable="true"]').forEach(el => el.addEventListener('input', handleTextEdit));
}


function renderPreviewImage(container: Element, step: LessonStep) {
    container.innerHTML = ''; // Clear existing content
    if (step.image.base64Data) {
        const img = document.createElement('img');
        img.src = `data:image/jpeg;base64,${step.image.base64Data}`;
        img.alt = step.image.prompt;
        container.appendChild(img);
    } else if (step.image.required) {
        const placeholderDiv = document.createElement('div');
        placeholderDiv.className = 'image-placeholder';
        setElementInnerHtml(placeholderDiv, sanitizeHtml(`
            <i class="fa-solid fa-image"></i>
            <strong>Image Placeholder</strong>
            <p><strong>Prompt:</strong> ${step.image.prompt}</p>
            <small>Recommended aspect ratio: 4:3</small>
        `));
        container.appendChild(placeholderDiv);
    }
}


function handleTextEdit(event: Event) {
    const target = event.target as HTMLElement;
    const fieldPath = target.dataset.field!;
    const stepAttr = target.dataset.step;
    const indexAttr = target.dataset.index;
    const lang = target.dataset.lang; // For bilingual fields
    const shouldAutoTranslate = lang === 'en' && languageSelect.value !== 'English' && fieldPath !== 'image.prompt';
    const shouldCancelAutoTranslation = lang === 'targetLang';

    if (stepAttr) {
        audioCache.delete(`step-${parseInt(stepAttr, 10)}-full-audio`);
    } else if (fieldPath === 'engagingQuestion') {
        audioCache.delete('engaging-question-full-audio');
    }

    const value = target.innerHTML; // Always use innerHTML for rich text

    if (fieldPath === 'keyTakeaways' && indexAttr) {
        audioCache.delete('summary-full-audio');
        const index = parseInt(indexAttr, 10);
        const takeaway = currentLessonData!.keyTakeaways[index];
        if (lang && typeof takeaway === 'object') {
            (takeaway as BilingualText)[lang as keyof BilingualText] = target.innerText; // no rich text in takeaways
        } else {
            currentLessonData!.keyTakeaways[index] = target.innerText;
        }
        if (shouldCancelAutoTranslation) {
            cancelPendingAutoTranslation(fieldPath, stepAttr, indexAttr, target);
        } else if (shouldAutoTranslate) {
            scheduleAutoTranslationFromEnglishEdit(target, fieldPath, stepAttr, indexAttr);
        }
        saveAppState();
        return;
    }

    if (!stepAttr) { // Lesson-level field
        updateNestedValue(currentLessonData, fieldPath, value, lang as keyof BilingualText | undefined);
        if (shouldCancelAutoTranslation) {
            cancelPendingAutoTranslation(fieldPath, stepAttr, indexAttr, target);
        } else if (shouldAutoTranslate) {
            scheduleAutoTranslationFromEnglishEdit(target, fieldPath, stepAttr, indexAttr);
        }
        saveAppState();
        return;
    }

    const stepIndex = parseInt(stepAttr, 10) - 1;
    if (!currentLessonData || !currentLessonData.steps[stepIndex]) return;

    updateNestedValue(currentLessonData.steps[stepIndex], fieldPath, value, lang as keyof BilingualText | undefined);
    if (shouldCancelAutoTranslation) {
        cancelPendingAutoTranslation(fieldPath, stepAttr, indexAttr, target);
    } else if (shouldAutoTranslate) {
        scheduleAutoTranslationFromEnglishEdit(target, fieldPath, stepAttr, indexAttr);
    }
    saveAppState();
}

function handleDeleteStep(indexToDelete: number) {
    if (!currentLessonData || !currentLessonData.steps) return;
    if (indexToDelete < 0 || indexToDelete >= currentLessonData.steps.length) return;

    // Create a new array of steps, excluding the one at the specified index.
    const newSteps = currentLessonData.steps.filter((_, index) => index !== indexToDelete);

    // Re-number the 'step' property of the remaining steps.
    newSteps.forEach((step, index) => {
        step.step = index + 1;
    });

    // Update the lesson data with the new array of steps.
    currentLessonData.steps = newSteps;

    // Re-render the entire preview to reflect the deletion and re-numbering.
    renderPreview(currentLessonData);
    saveAppState();
}

function handleDeleteTakeaway(indexToDelete: number) {
    if (currentLessonData && currentLessonData.keyTakeaways) {
        if (indexToDelete >= 0 && indexToDelete < currentLessonData.keyTakeaways.length) {
            currentLessonData.keyTakeaways.splice(indexToDelete, 1);
            renderPreview(currentLessonData);
            saveAppState();
        }
    }
}

function handleMoveStep(index: number, direction: string) {
    if (!currentLessonData || !currentLessonData.steps) return;

    if (direction === 'up' && index > 0) {
        [currentLessonData.steps[index], currentLessonData.steps[index - 1]] = 
        [currentLessonData.steps[index - 1], currentLessonData.steps[index]];
    } else if (direction === 'down' && index < currentLessonData.steps.length - 1) {
        [currentLessonData.steps[index], currentLessonData.steps[index + 1]] = 
        [currentLessonData.steps[index + 1], currentLessonData.steps[index]];
    }

    currentLessonData.steps.forEach((step, i) => {
        step.step = i + 1;
    });

    renderPreview(currentLessonData);
    saveAppState();
}


function handleImageReplace(event: Event) {
    const target = event.target as HTMLElement;
    const stepIndex = parseInt(target.dataset.step!, 10) - 1;

    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'image/*';
    fileInput.onchange = async () => {
        if (fileInput.files && fileInput.files[0]) {
            const file = fileInput.files[0];
            const { base64, mimeType } = await fileToBase64(file);
            const compressedBase64 = await compressImage(base64, mimeType);

            if (currentLessonData) {
                const step = currentLessonData.steps[stepIndex];
                step.image.base64Data = compressedBase64;
                step.image.required = true; // Ensure it's now considered required
                step.image.fileName = `step_${step.step}.jpg`;
                
                // PERFORMANCE: Update only the changed image instead of re-rendering everything
                const imageContainer = document.querySelector(`#image-container-${step.step}`);
                if (imageContainer) {
                    renderPreviewImage(imageContainer, step);
                }
                saveAppState();
            }
        }
    };
    fileInput.click();
}

/**
 * Toggles the visibility of the image edit controls for a specific step.
 */
function toggleImageEditControls(event: Event) {
    const target = event.currentTarget as HTMLElement;
    const step = target.dataset.step!;
    const editControls = document.getElementById(`image-edit-controls-${step}`);
    if (editControls) {
        const isVisible = editControls.style.display === 'flex';
        editControls.style.display = isVisible ? 'none' : 'flex';
    }
}

/**
 * Handles the AI image editing process.
 */
async function handleImageEdit(event: Event) {
    const target = event.currentTarget as HTMLButtonElement;
    const stepIndex = parseInt(target.dataset.step!, 10) - 1;

    const editContainer = document.getElementById(`image-edit-controls-${stepIndex + 1}`);
    const promptInput = editContainer?.querySelector('.image-edit-prompt') as HTMLInputElement;
    const prompt = promptInput?.value.trim();

    if (!prompt) {
        showError("Please enter a prompt to edit the image.");
        return;
    }
    if (!currentLessonData || !currentLessonData.steps[stepIndex]?.image.base64Data) {
        showError("No image found to edit.");
        return;
    }
    
    clearError();

    const imageContainer = document.querySelector(`#image-container-${stepIndex + 1}`);
    const imageWrapper = imageContainer?.parentElement;
    if (!imageWrapper) return;

    // Show loader
    const loader = document.createElement('div');
    loader.className = 'image-edit-loader';
    loader.innerHTML = `<div class="spinner"></div>`;
    imageWrapper.appendChild(loader);
    target.disabled = true;
    (target.previousElementSibling as HTMLInputElement).disabled = true; // Disable input
    
    try {
        const step = currentLessonData.steps[stepIndex];
        const originalBase64 = step.image.base64Data!;

        const imageResponse = await ai.models.generateContent({
            model: 'gemini-2.5-flash-image',
            contents: {
                parts: [
                    { inlineData: { data: originalBase64, mimeType: 'image/jpeg' } },
                    { text: prompt },
                ],
            },
            config: {
                responseModalities: [Modality.IMAGE],
            },
        });

        let newRawBase64 = '';
        if (imageResponse.candidates && imageResponse.candidates[0].content && imageResponse.candidates[0].content.parts) {
            for (const part of imageResponse.candidates[0].content.parts) {
                if (part.inlineData) {
                    newRawBase64 = part.inlineData.data;
                    break;
                }
            }
        }

        if (!newRawBase64) {
            throw new Error("The AI did not return a new image. Please try a different prompt.");
        }
        
        // The model returns a PNG, compress it to JPEG
        const compressedBase64 = await compressImage(newRawBase64, 'image/png');
        step.image.base64Data = compressedBase64;
        step.image.fileName = `step_${step.step}.jpg`;
        saveAppState();

        // Re-render the image preview
        if (imageContainer) {
            renderPreviewImage(imageContainer, step);
        }

        // Hide the edit controls
        if (editContainer) {
          editContainer.style.display = 'none';
          promptInput.value = '';
        }

    } catch (error) {
        console.error("Image editing failed:", error);
        showError(getFriendlyErrorMessage("Image editing failed", error));
    } finally {
        // Re-enable controls and remove loader
        loader.remove();
        target.disabled = false;
        (target.previousElementSibling as HTMLInputElement).disabled = false;
    }
}

/**
 * Handles regenerating an image from its edited prompt.
 */
async function handleImageRegenerate(event: Event) {
    const target = event.currentTarget as HTMLButtonElement;
    const stepIndex = parseInt(target.dataset.step!, 10) - 1;

    if (!currentLessonData || !currentLessonData.steps[stepIndex]) {
        showError("Could not find lesson data to regenerate image.");
        return;
    }
    
    clearError();

    const step = currentLessonData.steps[stepIndex];
    const prompt = step.image.prompt;

    if (!prompt) {
        showError("Please enter a prompt to generate an image.");
        return;
    }
    
    const imageContainer = document.querySelector(`#image-container-${step.step}`);
    const imageWrapper = imageContainer?.parentElement;
    if (!imageWrapper) return;

    // Show loader
    const loader = document.createElement('div');
    loader.className = 'image-edit-loader';
    loader.innerHTML = `<div class="spinner"></div>`;
    imageWrapper.appendChild(loader);
    target.disabled = true;

    try {
        const imageResponse = await ai.models.generateContent({
            model: 'gemini-2.5-flash-image',
            contents: {
                parts: [{ text: prompt }],
            },
            config: {
                responseModalities: [Modality.IMAGE],
            },
        });

        let newRawBase64 = '';
        if (imageResponse.candidates && imageResponse.candidates[0].content && imageResponse.candidates[0].content.parts) {
            for (const part of imageResponse.candidates[0].content.parts) {
                if (part.inlineData) {
                    newRawBase64 = part.inlineData.data;
                    break;
                }
            }
        }

        if (!newRawBase64) {
            throw new Error("The AI did not return an image. Please try a different prompt.");
        }
        
        const compressedBase64 = await compressImage(newRawBase64, 'image/png');
        step.image.base64Data = compressedBase64;
        step.image.required = true; // Make sure it's considered present
        step.image.fileName = `step_${step.step}.jpg`;
        saveAppState();

        // Re-render the image preview
        if (imageContainer) {
            renderPreviewImage(imageContainer, step);
        }

        // The image has been updated, so the 'Edit with AI' button should now be enabled.
        const editButton = document.querySelector(`.edit-image-btn[data-step="${step.step}"]`) as HTMLButtonElement;
        if (editButton) {
            editButton.disabled = false;
        }

    } catch (error) {
        console.error("Image regeneration failed:", error);
        showError(getFriendlyErrorMessage("Image regeneration failed", error));
        // If generation fails, we should clear the base64 data to show the placeholder again
        step.image.base64Data = '';
        if (imageContainer) {
            renderPreviewImage(imageContainer, step);
        }
        saveAppState();

    } finally {
        // Re-enable controls and remove loader
        loader.remove();
        target.disabled = false;
    }
}


// --- WAV FILE GENERATION ---

/**
 * Helper to write a string to a DataView.
 */
function writeString(view: DataView, offset: number, string: string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

/**
 * Creates a WAV file Blob from raw PCM audio data.
 * @param pcmData The raw audio data (Int16).
 * @param sampleRate The sample rate of the audio (e.g., 24000).
 * @returns A Blob representing the WAV file.
 */
function createWavFileBlob(pcmData: Uint8Array, sampleRate: number): Blob {
    const numChannels = 1;
    const bitsPerSample = 16;
    const dataSize = pcmData.length;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);

    // RIFF chunk descriptor
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(view, 8, 'WAVE');

    // FMT sub-chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true); // Subchunk1Size for PCM
    view.setUint16(20, 1, true); // AudioFormat (1=PCM)
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numChannels * (bitsPerSample / 8), true); // ByteRate
    view.setUint16(32, numChannels * (bitsPerSample / 8), true); // BlockAlign
    view.setUint16(34, bitsPerSample, true);

    // DATA sub-chunk
    writeString(view, 36, 'data');
    view.setUint32(40, dataSize, true);

    // Write PCM data
    new Uint8Array(buffer, 44).set(pcmData);

    return new Blob([view], { type: 'audio/wav' });
}

function getMonolingualLesson(lesson: Lesson): any {
    const convert = (value: any): any => {
        // Base case: if it's not an object, or it's null, return it directly.
        if (typeof value !== 'object' || value === null) {
            return value;
        }

        // If it's an array, map over its elements and convert each one.
        if (Array.isArray(value)) {
            return value.map(convert);
        }

        // If it's a bilingual text object, return only the target language string.
        if (Object.prototype.hasOwnProperty.call(value, 'en') && Object.prototype.hasOwnProperty.call(value, 'targetLang')) {
            return value.targetLang;
        }

        // For any other object, create a new object and convert each of its properties.
        const newObj: { [key: string]: any } = {};
        for (const key in value) {
            if (Object.prototype.hasOwnProperty.call(value, key)) {
                newObj[key] = convert(value[key]);
            }
        }
        return newObj;
    };
    // Deep copy to avoid modifying the original lesson data used in the preview.
    return convert(JSON.parse(JSON.stringify(lesson)));
}


/**
 * Creates and downloads a ZIP file of the lesson, now including audio files.
 */
async function createAndDownloadZip() {
  if (!currentLessonData) return;

  const originalButtonHtml = downloadBtn.innerHTML;
  downloadBtn.disabled = true;
  generateBtn.disabled = true;

  try {
    const isBilingual = languageSelect.value !== 'English';
    const lessonForExport = isBilingual ? getMonolingualLesson(currentLessonData) : currentLessonData;

    const zip = new JSZip();
    const imageFolder = zip.folder("images");
    const audioFolder = zip.folder("audio");
    const packagedImageFileNames = new Set<string>();

    // Add images to zip
    for (const step of currentLessonData.steps) {
      if (step.image.base64Data && step.image.fileName) {
        imageFolder!.file(step.image.fileName, step.image.base64Data, { base64: true });
        packagedImageFileNames.add(step.image.fileName);
      }
    }

    // --- Generate one audio file per page ---
    const audioTasks: { id: string; text: string }[] = [];
    const tempDiv = document.createElement('div');
    const getPlainTextForAudio = (text: string): string => {
        tempDiv.innerHTML = text;
        return tempDiv.innerText || text;
    };
    const getTextForAudio = (data: string | BilingualText): string => {
        if (typeof data === 'string') {
            return data;
        }
        if (typeof data === 'object' && data !== null) {
            if (isBilingual) {
                return (data as BilingualText).targetLang || '';
            }
            return (data as BilingualText).en || '';
        }
        return '';
    };

    if (currentLessonData.engagingQuestion) {
        const titleText = getTextForAudio(currentLessonData.title);
        const questionText = getTextForAudio(currentLessonData.engagingQuestion);
        audioTasks.push({ id: 'engaging-question', text: `${titleText}. ${questionText}` });
    }
    currentLessonData.steps.forEach(step => {
        const cleanTitle = getPlainTextForAudio(getTextForAudio(step.title));
        const cleanExplanation = getPlainTextForAudio(getTextForAudio(step.explanation));
        const cleanHint = getPlainTextForAudio(getTextForAudio(step.nextStepHint));

        const combinedText = [
            cleanTitle,
            cleanExplanation,
            cleanHint
        ].filter(text => text && text.trim().length > 0).join('. ');

        if (combinedText) {
            audioTasks.push({ id: `step-${step.step}`, text: combinedText });
        }
    });

    // Add summary audio tasks
    const summaryTitleText = getTextForAudio(currentLessonData.uiTranslations.lessonSummary);
    const keyTakeawaysTitleText = getTextForAudio(currentLessonData.uiTranslations.keyTakeaways);
    
    const SUMMARY_PAGE_SIZE = 5;
    const takeaways = currentLessonData.keyTakeaways;
    const totalPages = Math.ceil(takeaways.length / SUMMARY_PAGE_SIZE);

    for (let i = 0; i < totalPages; i++) {
        const startIndex = i * SUMMARY_PAGE_SIZE;
        const endIndex = Math.min(startIndex + SUMMARY_PAGE_SIZE, takeaways.length);
        const pageTakeaways = takeaways.slice(startIndex, endIndex);
        
        const takeawaysText = pageTakeaways.map(getTextForAudio).join('. ');
        
        // Only add title to the first page audio
        const headerText = (i === 0) ? `${summaryTitleText}. ${keyTakeawaysTitleText}.` : '';
        const summaryText = [headerText, takeawaysText].filter(Boolean).join(' ');

        if (summaryText.trim()) {
            audioTasks.push({ id: `summary-${i + 1}`, text: summaryText });
        }
    }

    let generatedCount = 0;
    const failedAudioTasks: string[] = [];
    const successfulAudioTasks: { id: string }[] = [];

    for (const task of audioTasks) {
        generatedCount++;
        downloadBtn.innerHTML = `
            <i class="fa-solid fa-spinner fa-spin"></i> Generating Audio (${generatedCount}/${audioTasks.length})
        `;
        try {
            const base64Audio = await getTtsAudio(task.text);
            if (base64Audio) {
                const pcmData = decode(base64Audio);
                const wavBlob = createWavFileBlob(pcmData, 24000);
                audioFolder!.file(`${task.id}.wav`, wavBlob);
                successfulAudioTasks.push({ id: task.id });
            } else {
                throw new Error("API returned no audio data.");
            }
        } catch(err) {
            console.warn(`Could not generate audio for "${task.id}":`, err);
            failedAudioTasks.push(task.id);
            // Continue even if one audio file fails
        }
    }
    
    updateUiWithFailedAudioLinks(failedAudioTasks);
    
    downloadBtn.innerHTML = `
      <i class="fa-solid fa-spinner fa-spin"></i> Packaging Files...
    `;

    // Add HTML file
    const htmlContent = generateLessonHtml(lessonForExport, successfulAudioTasks, packagedImageFileNames);
    zip.file("index.html", htmlContent);
    
    // Generate and download zip
    const zipBlob = await zip.generateAsync({ type: "blob" });
    const url = URL.createObjectURL(zipBlob);
    const a = document.createElement("a");
    a.href = url;
    const downloadBaseName = toSafeAsciiFileName(getTextValue(lessonForExport.title), 'lesson');
    a.download = `${downloadBaseName}_lesson.zip`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (error) {
    console.error("Failed to create ZIP:", error);
    showError(getFriendlyErrorMessage("Could not create the lesson package", error));
  } finally {
    // Restore the button after a short delay
    setTimeout(() => {
        downloadBtn.disabled = false;
        generateBtn.disabled = false;
        downloadBtn.innerHTML = originalButtonHtml;
    }, 500);
  }
}

// --- HTML GENERATION FOR DOWNLOAD ---

function getLessonCss() {
  return `
        :root { --accent: #386AF6; --bg: #f4f7f9; --secondary-bg: #e8eef3; --panel-bg: #fff; --text: #333; --border: #d1d9e0; --button: #FCB717; --button-hover: #e0a810; --highlight: #CD0FD5; --success: #28a745; --danger: #dc3545; --text-secondary: #555; }
        body, html { margin: 0; padding: 0; width: 100%; height: 100%; overflow: hidden; font-family: -apple-system, BlinkMacFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: var(--bg); color: var(--text); font-size: 16px; }
        #start-screen { display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; height: 100vh; background: linear-gradient(100deg, #6253E1, #A44BC5); color: white; padding: 2rem; box-sizing: border-box; }
        #start-screen h1 { font-size: clamp(2.5rem, 5vw, 4rem); margin-bottom: 1rem; text-shadow: 2px 2px 8px rgba(0,0,0,0.3); }
        #start-screen .engaging-question { max-width: 800px; margin: 1rem 0; font-style: italic; font-size: 1.4rem; line-height: 1.4; text-shadow: 1px 1px 4px rgba(0,0,0,0.4); }
        #start-btn { background-color: var(--button); color: #333; border: none; border-radius: 50px; padding: 1.2rem 3rem; font-size: 1.4rem; font-weight: bold; cursor: pointer; transition: all 0.3s; box-shadow: 0 4px 15px rgba(0,0,0,0.3); display: flex; align-items: center; gap: 0.75rem; }
        #start-btn:hover { transform: scale(1.05); background-color: var(--button-hover); }
        #app { background: white; width: 100vw; height: 100vh; overflow: hidden; position: relative; }
        .lesson-content { width: 100%; height: 100%; overflow: hidden; }
        .step-view { display: none; }
        .step-view.active { 
            display: flex; 
            gap: 3rem; 
            padding: 2rem 4rem;
            width: 100%;
            height: 100%;
            box-sizing: border-box;
            background: white;
            animation: fadeIn 0.4s ease-out;
            overflow: hidden;
            align-items: center;
        }
        .step-image-container { flex: 1.2; height: 80%; display: flex; align-items: center; justify-content: center; background: #fafafa; border-radius: 12px; border: 1px solid #eee; overflow: hidden; }
        .step-image-container img { max-width: 100%; max-height: 100%; object-fit: contain; }
        .step-content-container { flex: 1; height: 80%; display: flex; flex-direction: column; justify-content: flex-start; gap: 1.5rem; overflow-y: auto; padding-right: 10px; }
        .step-content-container::-webkit-scrollbar { width: 6px; }
        .step-content-container::-webkit-scrollbar-thumb { background: #ccc; border-radius: 10px; }
        .step-content-container div { font-size: 1.2rem; line-height: 1.6; text-align: justify; }
        .floating-back-btn {
            position: absolute;
            left: 15px;
            top: 50%;
            transform: translateY(-50%);
            width: 40px;
            height: 40px;
            background-color: white;
            border: 1px solid var(--border);
            border-radius: 50%;
            display: none;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            z-index: 100;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            color: var(--text-secondary);
            transition: all 0.2s;
        }
        .floating-back-btn:hover { background-color: var(--secondary-bg); color: var(--accent); transform: translateY(-50%) scale(1.1); }
        .floating-back-btn i { font-size: 1.1rem; }
        .audio-player-btn { display: none !important; }
        .key-term { color: var(--accent); font-weight: 700; border-bottom: 2px solid #e0e7ff; }
        .next-step-hint { 
            display: flex; 
            align-items: center; 
            gap: 1.5rem; 
            background-color: #f0f7ff; 
            padding: 1.5rem; 
            border-left: 6px solid var(--accent); 
            border-radius: 8px; 
            font-style: italic; 
            font-size: 1.2rem; 
            margin-top: 2rem;
        }
        .next-step-hint i { color: var(--accent); font-size: 1.6rem; flex-shrink: 0; }
        .next-step-hint p { margin: 0; text-align: justify; line-height: 1.5; }
        .tf-buttons, .mcq-options { display: flex; flex-direction: column; gap: 0.75rem; }
        .tf-btn { width: 100%; text-align: left; padding: 1rem; background: white; border: 1px solid var(--border); border-radius: 8px; cursor: pointer; transition: 0.2s; font-size: 1.1rem; font-weight: 500; }
        .tf-btn:hover:not(:disabled) { background: #f8faff; border-color: var(--accent); }
        .tf-btn.correct { background: var(--success); color: white; border-color: var(--success); }
        .tf-btn.incorrect { background: var(--danger); color: white; border-color: var(--danger); }
        .feedback { margin-top: 1rem; font-weight: bold; text-align: center; font-size: 1.1rem; border-radius: 4px; padding: 0.5rem; }
        .feedback.correct { color: var(--success); background: #e9f7ec; }
        .feedback.incorrect { color: var(--danger); background: #fbebee; }
        .continue-btn { background: var(--accent); color: white; border: none; border-radius: 50px; padding: 1rem 2rem; font-size: 1.1rem; font-weight: bold; cursor: pointer; margin-top: 1rem; align-self: center; box-shadow: 0 4px 10px rgba(56,106,246,0.3); }
        .continue-btn:hover { transform: translateY(-2px); }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes shake { 0%, 100% { transform: translateX(0); } 25% { transform: translateX(-5px); } 75% { transform: translateX(5px); } }
        .shake-incorrect { animation: shake 0.4s ease-in-out; }
        .completion-screen { text-align: center; height: 100%; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 2rem; background: linear-gradient(135deg, #fff 0%, #f0f7ff 100%); }
        .completion-screen h2 { font-size: 3rem; color: var(--accent); margin: 0; }
        .completion-screen p { font-size: 1.5rem; color: #555; }
        .key-takeaways-summary { 
            text-align: left; 
            padding: 1rem 0; 
            width: 100%; 
            max-width: 900px; 
            height: auto; 
            overflow: visible; 
        }
        .key-takeaways-summary ul { list-style: none; padding: 0; }
        .key-takeaways-summary li { 
            padding: 1.2rem 0; 
            border-bottom: 1px solid #eee; 
            display: flex; 
            align-items: flex-start; 
            gap: 1.2rem; 
            font-size: 1.3rem; 
            line-height: 1.4;
            text-align: justify;
        }
        .key-takeaways-summary li::before { content: '\\f058'; font-family: 'Font Awesome 6 Free'; font-weight: 900; color: var(--success); flex-shrink: 0; margin-top: 4px; }
    `;
}

function getLessonScript(lessonJsonString: string) {
    return `
    const lessonData = ${lessonJsonString};
    const ui = lessonData.uiTranslations || {};
    let currentStep = -1; // -1 means start screen
    let currentSummaryPage = 0;
    const SUMMARY_PAGE_SIZE = 5;
    const contentEl = document.getElementById('lesson-content');
    const startScreen = document.getElementById('start-screen');
    const appContainer = document.getElementById('app');
    const startBtn = document.getElementById('start-btn');
    let currentAudio = null;
    let completionTimer = null;

    const GENERATED_INDIC_CHAR_RANGES = '\\u0900-\\u097F\\u0980-\\u09FF\\u0A80-\\u0AFF\\u0C00-\\u0C7F';
    const GENERATED_INDIC_TEXT_REGEX = new RegExp('[' + GENERATED_INDIC_CHAR_RANGES + ']');

    function wrapIndicTextInLatex(mathContent) {
        if (!GENERATED_INDIC_TEXT_REGEX.test(mathContent)) return mathContent;

        const protectedTextCommands = [];
        let normalized = mathContent.replace(/\\\\(?:text|mbox|mathrm)\\{[^{}]*\\}/g, function(match) {
            const index = protectedTextCommands.push(match) - 1;
            return '@@TEXT_CMD_' + index + '@@';
        });

        const indicInBraceGroup = new RegExp('\\\\{([^{}]*[' + GENERATED_INDIC_CHAR_RANGES + '][^{}]*)\\\\}', 'g');
        normalized = normalized.replace(indicInBraceGroup, function(match, group) {
            if (group.indexOf('\\\\') !== -1) return match;
            const trimmed = group.trim();
            return trimmed ? '{\\\\text{' + trimmed + '}}' : match;
        });

        const indicTextNearOperator = new RegExp('(^|[=+\\\\-*/^_(),:;])\\\\s*([' + GENERATED_INDIC_CHAR_RANGES + '][' + GENERATED_INDIC_CHAR_RANGES + '\\\\s\\\\u200c\\\\u200d\\\\u0964.,\\'"!?-]*[' + GENERATED_INDIC_CHAR_RANGES + '])(?=\\\\s*($|[=+\\\\-*/^_(),:;]))', 'g');
        normalized = normalized.replace(indicTextNearOperator, function(match, prefix, phrase) {
            const trimmed = phrase.trim();
            return trimmed ? prefix + '\\\\text{' + trimmed + '}' : match;
        });

        return normalized.replace(/@@TEXT_CMD_(\\d+)@@/g, function(_, index) {
            return protectedTextCommands[Number(index)] || '';
        });
    }

    function normalizeMultilingualLatex(text) {
        if (!text || !GENERATED_INDIC_TEXT_REGEX.test(text) || text.indexOf('$') === -1) return text;
        return text.replace(/(\\$\\$?)([\\s\\S]*?)\\1/g, function(match, delimiter, mathContent) {
            return delimiter + wrapIndicTextInLatex(mathContent) + delimiter;
        });
    }

    function renderFormattedText(value) {
        return normalizeMultilingualLatex(String(value || ''))
            .replace(/\\|\\|\\|(.*?)\\|\\|\\|/g, '<strong class="key-term">$1</strong>')
            .replace(/\\n/g, '<br>');
    }

    function typesetMath() {
        if (window.MathJax && window.MathJax.typesetPromise) {
            window.MathJax.typesetPromise([contentEl]).catch(function(error) {
                console.log('MathJax typeset failed:', error);
            });
        }
    }

    const floatingBackBtn = document.createElement('button');
    floatingBackBtn.className = 'floating-back-btn';
    floatingBackBtn.innerHTML = '<i class="fa-solid fa-chevron-left"></i>';
    floatingBackBtn.style.display = 'none';
    appContainer.appendChild(floatingBackBtn);
    floatingBackBtn.onclick = goToPreviousStep;

    function stopAllAudio() {
        if (completionTimer) {
            clearTimeout(completionTimer);
            completionTimer = null;
        }
        if (currentAudio) {
            currentAudio.pause();
            currentAudio.currentTime = 0;
            currentAudio = null;
        }
        document.querySelectorAll('audio').forEach(a => {
            a.pause();
            a.currentTime = 0;
        });
    }

    function handleAudioComplete(id) {
        if (completionTimer) clearTimeout(completionTimer);
        completionTimer = setTimeout(function() {
            completionTimer = null;
            if (id === 'player-engaging-question' && currentStep < 0) {
                startLesson();
            } else if (id.startsWith('player-step-')) {
                goToNextStep();
            } else if (id.startsWith('player-summary-')) {
                goToNextStep();
            }
        }, 2000);
    }

    function playAudio(id) {
        console.log('Playing audio:', id);
        stopAllAudio();
        const player = document.getElementById(id);
        if (!player) {
            console.log('Missing audio file, continuing without it:', id);
            handleAudioComplete(id);
            return false;
        }

        player.currentTime = 0;
        currentAudio = player;
        const playPromise = player.play();
        if (playPromise && typeof playPromise.catch === 'function') {
            playPromise.catch(function(error) {
                console.log('Audio playback failed, continuing without it:', id, error);
                handleAudioComplete(id);
            });
        }
        return true;
    }

    function goToPreviousStep() {
        if (currentSummaryPage > 0) {
            currentSummaryPage--;
            renderSummaryScreen();
        } else if (currentStep > 0) {
            if (currentStep === lessonData.steps.length) {
                // Was on first summary page, go back to last step
                currentStep--;
                renderStep();
            } else {
                currentStep--;
                renderStep();
            }
        } else if (currentStep === 0) {
            currentStep = -1;
            appContainer.style.display = 'none';
            startScreen.style.display = 'flex';
            floatingBackBtn.style.display = 'none';
            startBtn.style.display = '';
            stopAllAudio();
        }
    }

    function goToNextStep() {
        if (currentStep < lessonData.steps.length - 1) {
            currentStep++;
            renderStep();
        } else {
            const totalSummaryPages = Math.ceil(lessonData.keyTakeaways.length / SUMMARY_PAGE_SIZE);
            if (currentStep === lessonData.steps.length - 1) {
                 // Transitioning to summary mode
                 currentStep = lessonData.steps.length;
                 currentSummaryPage = 0;
                 renderSummaryScreen();
            } else if (currentSummaryPage < totalSummaryPages - 1) {
                 currentSummaryPage++;
                 renderSummaryScreen();
            } else {
                renderCompletionScreen();
            }
        }
    }

    function renderStep() {
        const step = lessonData.steps[currentStep];
        floatingBackBtn.style.display = currentStep > 0 ? 'flex' : 'none';

        const imageHtml = step.image.base64Data ? \`<img src="\${step.image.base64Data}" alt="\${step.image.prompt}">\`
            : (step.image.required ? \`<div class="image-placeholder"><strong>Image Placeholder</strong><p><strong>Prompt:</strong> \${step.image.prompt}</p></div>\` : '');
        const explanationHtml = renderFormattedText(step.explanation);
        const nextStepHintHtml = step.nextStepHint ? \`<div class="next-step-hint"><i class="fa-solid fa-arrow-right-long"></i> <p>\${renderFormattedText(step.nextStepHint)}</p></div>\` : '';
        
        contentEl.innerHTML = \`
            <div class="step-view active">
                <div class="step-image-container">\${imageHtml}</div>
                <div class="step-content-container">
                    <div>
                        <h2>\${renderFormattedText(step.title)}</h2>
                        <div>\${explanationHtml}</div>
                        \${nextStepHintHtml}
                    </div>
                </div>
            </div>\`;
        
        typesetMath();
        playAudio(\`player-step-\${step.step}\`);
    }

    // Start sequence on button click
    startBtn.onclick = () => {
        startLesson();
    };

    function startLesson() {
        startScreen.style.display = 'none';
        appContainer.style.display = 'block';
        currentStep = 0;
        renderStep();
    }

    // Auto-advance logic
    const players = document.querySelectorAll('audio');
    players.forEach(p => {
        p.onended = () => handleAudioComplete(p.id);
        p.onerror = () => handleAudioComplete(p.id);
    });

    function renderSummaryScreen() {
        floatingBackBtn.style.display = 'flex';
        const totalPages = Math.ceil(lessonData.keyTakeaways.length / SUMMARY_PAGE_SIZE);
        
        const startIndex = currentSummaryPage * SUMMARY_PAGE_SIZE;
        const pageTakes = lessonData.keyTakeaways.slice(startIndex, startIndex + SUMMARY_PAGE_SIZE);
        const takesHtml = pageTakes.map(t => '<li>' + renderFormattedText(t) + '</li>').join('');
        
        const title = renderFormattedText(ui.lessonSummary || 'Summary');
        const buttonText = (currentSummaryPage < totalPages - 1) ? (ui.next || 'Next') : (ui.finishLesson || 'Finish');
        
        const pageIndicator = totalPages > 1 ? '<div style="font-size: 0.9rem; color: #999; margin-top: 1rem;">Page ' + (currentSummaryPage + 1) + ' of ' + totalPages + '</div>' : '';

        contentEl.innerHTML = \`
            <div class="step-view active" style="flex-direction: column; justify-content: center; align-items: center; padding: 2rem; overflow: hidden;">
                <div style="width: 100%; max-width: 900px; display: flex; flex-direction: column; align-items: center; height: auto;">
                    <h2 style="font-size: 2.5rem; color: var(--accent); margin-bottom: 2rem; margin-top: 0;">\${title}</h2>
                    <div class="key-takeaways-summary" style="height: auto; max-height: none; overflow: visible; box-shadow: none; border: none; background: transparent; padding: 0;">
                        <ul>\${takesHtml}</ul>
                    </div>
                    \${pageIndicator}
                    <button class="continue-btn" id="summary-next-btn" style="margin-top: 2rem;">\${buttonText} <i class="fa-solid fa-flag-checkered"></i></button>
                </div>
            </div>\`;
        
        document.getElementById('summary-next-btn').onclick = goToNextStep;
        typesetMath();
        playAudio('player-summary-' + (currentSummaryPage + 1));
    }

    function renderCompletionScreen() {
        floatingBackBtn.style.display = 'none';
        const completionTitleText = renderFormattedText(ui.completionTitle || ui.done || ui.finishLesson || 'Congratulations!');
        const completionMessageText = ui.completionMessage ? renderFormattedText(ui.completionMessage) : '';
        const lessonTitleText = renderFormattedText(lessonData.title);
        const completionMessageHtml = completionMessageText
            ? \`<p>\${completionMessageText} <strong>\${lessonTitleText}</strong></p>\`
            : \`<p><strong>\${lessonTitleText}</strong></p>\`;

        contentEl.innerHTML = \`
            <div class="completion-screen">
                <div class="confetti-container"></div>
                <div class="trophy-container">
                    <svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
                        <defs>
                            <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
                                <stop offset="0%" style="stop-color:#FFD700;stop-opacity:1" />
                                <stop offset="100%" style="stop-color:#FFA500;stop-opacity:1" />
                            </linearGradient>
                            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                                <feGaussianBlur stdDeviation="5" result="coloredBlur"/>
                                <feMerge>
                                    <feMergeNode in="coloredBlur"/>
                                    <feMergeNode in="SourceGraphic"/>
                                </feMerge>
                            </filter>
                        </defs>
                        <path d="M50,150 Q50,130 70,130 L130,130 Q150,130 150,150 L150,160 Q150,180 130,180 L70,180 Q50,180 50,160 Z" fill="#A0522D"/>
                        <rect x="85" y="120" width="30" height="10" fill="#A0522D"/>
                        <path d="M70,50 C40,50 40,120 70,120 L130,120 C160,120 160,50 130,50 Q100,20 70,50 Z" fill="url(#grad1)" filter="url(#glow)"/>
                        <path d="M60,60 C30,60 30,110 60,110" fill="none" stroke="url(#grad1)" stroke-width="5"/>
                        <path d="M140,60 C170,60 170,110 140,110" fill="none" stroke="url(#grad1)" stroke-width="5"/>
                    </svg>
                </div>
                <h2>\${completionTitleText}</h2>
                \${completionMessageHtml}
            </div>\`;
        typesetMath();
    }

    // Removed automatic audio initialization to favor button click trigger
    // Fallback listeners removed as per request for button-triggered sequence
    `;
}


function generateLessonHtml(
  lesson: Lesson,
  audioTasks: { id: string }[],
  packagedImageFileNames: Set<string> = new Set()
): string {
  // FIX: The `lesson` object passed here has been processed to be monolingual
  // (all BilingualText fields are strings). Cast to `any` to reflect this
  // and prevent type errors and bugs where an object might be rendered as text.
  const monolingualLesson = normalizeLessonMathSyntax(lesson as any);

  const lessonForHtml = JSON.parse(JSON.stringify(monolingualLesson));
  // Replace base64 with relative paths for the final HTML
  lessonForHtml.steps.forEach((step: LessonStep) => {
    step.image.base64Data = (step.image.fileName && packagedImageFileNames.has(step.image.fileName))
        ? `images/${step.image.fileName}` 
        : '';
  });

  // By replacing `<` with its unicode escape sequence, we prevent any stray `</script>`
  // tags within the lesson data from prematurely closing the script block in the browser.
  const lessonJsonString = JSON.stringify(lessonForHtml).replace(/</g, '\\u003c');
  
  const audioElementsHtml = audioTasks
    .map(task => `<audio id="player-${task.id}" src="audio/${task.id}.wav" preload="auto"></audio>`)
    .join('');
    
  const engagingQuestionAudioBtn = `<button class="audio-player-btn" data-player-id="player-engaging-question" title="Read Aloud" style="display:none;"><i class="fa-solid fa-volume-high"></i></button>`;
  const ui = monolingualLesson.uiTranslations || {};
  const letsExploreText = ui.letsExplore || "Let's Explore!";
  const documentLang = getHtmlLangCode(languageSelect.value);

  return `<!DOCTYPE html>
<html lang="${documentLang}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${monolingualLesson.title}</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
    <script>
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$']],
                displayMath: [['$$', '$$']]
            },
            chtml: {
                mtextInheritFont: true
            },
            options: {
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
            }
        };
    </script>
    <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
    <style>${getLessonCss()}</style>
</head>
<body>
<div id="start-screen">
    <h1>${monolingualLesson.title}</h1>
    <div class="engaging-question"><p>${monolingualLesson.engagingQuestion}</p>${engagingQuestionAudioBtn}</div>
    <button id="start-btn">${letsExploreText} <i class="fa-solid fa-rocket"></i></button>
</div>
<div id="app" style="display:none;">
    <main id="lesson-content" class="lesson-content"></main>
    <div id="audio-players" style="display: none;">${audioElementsHtml}</div>
</div>
<script>${getLessonScript(lessonJsonString)}<\/script>
</body>
</html>`;
}


// --- EVENT LISTENERS ---
generateBtn.addEventListener('click', generateLesson);
downloadBtn.addEventListener('click', createAndDownloadZip);
topicInput.addEventListener('keydown', (e) => {
    if(e.key === 'Enter') generateLesson();
});

// Event delegation for dynamic controls in the preview panel
lessonPreview.addEventListener('click', (event) => {
    const target = event.target as HTMLElement;

    // Handle step deletion
    const deleteStepBtn = target.closest('.delete-step-btn');
    if (deleteStepBtn) {
        const index = parseInt((deleteStepBtn as HTMLButtonElement).dataset.index!, 10);
        handleDeleteStep(index);
        return;
    }

    // Handle step movement
    const moveStepBtn = target.closest('.move-step-btn');
    if (moveStepBtn) {
        const index = parseInt((moveStepBtn as HTMLButtonElement).dataset.index!, 10);
        const direction = (moveStepBtn as HTMLButtonElement).dataset.direction!;
        handleMoveStep(index, direction);
        return;
    }

    // Handle takeaway deletion
    const deleteTakeawayBtn = target.closest('.delete-takeaway-btn');
    if (deleteTakeawayBtn) {
        const index = parseInt((deleteTakeawayBtn as HTMLButtonElement).dataset.index!, 10);
        handleDeleteTakeaway(index);
        return;
    }
});

// Update page count display when slider is moved
pagesInput.addEventListener('input', () => {
    pagesValue.textContent = pagesInput.value;
    saveAppState();
});

const inputsToSave = [
    topicInput,
    gradeSelect,
    subjectSelect,
    languageSelect,
    countrySelect,
    stateInput,
    specificPromptInput,
    pagesInput,
    videoLink
];

inputsToSave.forEach(input => {
    input.addEventListener('input', saveAppState);
    input.addEventListener('change', saveAppState);
});

loadAppState();

export {};
