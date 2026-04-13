# --- Step 1 / Step 2 Parameters ---
#
# This project has been reduced to the preprocessing stages only:
#   - Step 1 standardization
#   - Step 2 source separation


# --- Step-1 Parameters ---

STEP1_WAV_SUBTYPE = "PCM_16"


# --- Step-2 Parameters ---

STEP2_TARGET_SR = 44100
STEP2_WAV_SUBTYPE = "PCM_16"

# UVR / MDX-Net predictor params
STEP2_MODEL_DIM_F = 3072
STEP2_MODEL_DIM_T = 8
STEP2_MODEL_N_FFT = 6144
STEP2_MODEL_CHUNKS = 10
STEP2_MODEL_MARGIN = STEP2_TARGET_SR
STEP2_MODEL_DENOISE = False
