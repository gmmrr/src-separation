# --- NVV Pipeline – Global Constants ---
# constants.py remains reserved for true semantic constants only
# (naming keys, file extensions, structural identifiers).

# --- Pipeline Step Names ---
KEY_STEP_1 = "1_std"
KEY_STEP_2 = "2_sep"
KEY_STEP_3 = "3_norm"
KEY_STEP_4 = "4_vad"
KEY_STEP_5 = "5_asr"
KEY_STEP_6 = "6_nlp"
KEY_STEP_7 = "7_nvv"

# --- atomic suffix / naming tokens/key ---
KEY_ORIGINAL = "original"               # original file info in metadata
KEY_STD = "std"                         # step 1 output
KEY_VOCALS = "vocals"                   # step 2 output
KEY_BACKGROUND = "background"           # step 2 output
KEY_NORM = "norm"                       # Step 3 output
KEY_VAD = "vad"                         # Step 4 output
KEY_ASR = "asr"                         # Step 5 output
KEY_NLP = "nlp"                         # Step 6 output 
KEY_NVV = "nvv"                         # Step 7 output 
KEY_NVV_CANDIDATES = "nvv_candidates"   # Step 7 output annotation key
KEY_METADATA = "metadata"               # Metadata file

# --- metadata / path keys ---
KEY_AUDIO_ID = "audio_id"               # unique identifier for each audio file, used for folder naming and metadata linking
KEY_FILE = "file"                       # original file info: path, sr, channels etc.
KEY_AUDIO_FILES = "audios"              # processed audio files info: path, sr, channels etc.
KEY_ANNOTATIONS = "annotations"         # list of annotations and their properties
KEY_LABELS = "labels"                   # list of label files and their properties
KEY_STEP_LOG = "steps"                  # log of processing steps
KEY_PER_AUDIO = "per_audio"             # folders for each audio file, containing all processed files and annotations related to that audio file
KEY_GLOBAL = "global"                   # global-level evaluations and clips
KEY_CLIPS = "clips"                     # extracted clips for evaluation
KEY_EVALUATION = "evaluation"           # evaluation results and metrics
KEY_RQ = "research_questions"           # research question evaluation results

# --- Metadata Fields ---
KEY_FIELD_PATH = "path"
KEY_FIELD_SR = "sr"
KEY_FIELD_CHANNELS = "channels"

# --- Default extensions ---
EXT_WAV = ".wav"
EXT_JSON = ".json"
EXT_TXT = ".txt"


# --- Evaluation Modes ---
VALID_MODES = {"full_gt", "part_gt"}


# --- Audio Derivatives and VAD masks Single Source of Truth (please!) ---
'''
Definition of audio derivatives which are used as input for VAD and ASR 
and the VAD masks which are used for gating in ASR and NVV. 
These lists are used for validation and defaulting in load_pipeline_config.py, 
and they also serve as documentation of the available options.
Semantics:
- audio_derivatives: valid inputs for Step 4 (VAD generation) and Step 5 (ASR input).
- vad_masks: valid VAD mask selectors for Step 5 and Step 7; includes "no" to mean no VAD gating.
'''

# canonical audio derivative keys (used for metadata/yaml/validation)
AUDIO_ORIGINAL = KEY_ORIGINAL
AUDIO_STD = KEY_STD
AUDIO_STD_VOCALS = f"{AUDIO_STD}_{KEY_VOCALS}"
AUDIO_STD_BACKGROUND = f"{AUDIO_STD}_{KEY_BACKGROUND}"
AUDIO_STD_VOCALS_NORM = f"{AUDIO_STD_VOCALS}_{KEY_NORM}"
AUDIO_STD_BACKGROUND_NORM = f"{AUDIO_STD_BACKGROUND}_{KEY_NORM}"
VAD_NO = "no"

'''6 Audio Derivatives used as input for VAD (Step 4) and ASR (Step 5) - combined with all 7 VAD-masks.'''
AUDIO_DERIVATIVES = [
    AUDIO_ORIGINAL,
    AUDIO_STD,
    AUDIO_STD_VOCALS,
    AUDIO_STD_BACKGROUND,
    AUDIO_STD_VOCALS_NORM,
    AUDIO_STD_BACKGROUND_NORM,
]

'''7 VAD masks used as input for gating in Step 5 and Step 7 - combined with all 6 audio derivatives.'''
VAD_MASKS = [VAD_NO] + AUDIO_DERIVATIVES

