#!/usr/bin/env bash
set -e

echo "Creating model directories..."
mkdir -p models/external_repos
mkdir -p models/checkpoints/tfgridnet
mkdir -p models/checkpoints/usef_tse
mkdir -p models/checkpoints/eend_eda
echo "Done creating model directories."

echo ""
echo "Entering external repo directory..."
cd models/external_repos

echo ""
echo "Checking USEF-TSE repo..."
if [ ! -d "USEF-TSE" ]; then
  echo "Downloading USEF-TSE..."
  git clone https://github.com/ZBang/USEF-TSE.git
  echo "Done downloading USEF-TSE."
else
  echo "USEF-TSE already exists. Skipping."
fi

echo ""
echo "Checking EEND repo..."
if [ ! -d "EEND" ]; then
  echo "Downloading EEND..."
  git clone https://github.com/BUTSpeechFIT/EEND.git
  echo "Done downloading EEND."
else
  echo "EEND already exists. Skipping."
fi

echo ""
echo "Checking ESPnet repo..."
if [ ! -d "espnet" ]; then
  echo "Downloading ESPnet..."
  git clone https://github.com/espnet/espnet.git
  echo "Done downloading ESPnet."
else
  echo "ESPnet already exists. Skipping."
fi

echo ""
echo "Entering TF-GridNet checkpoint directory..."
cd ../checkpoints/tfgridnet

echo ""
echo "Checking TF-GridNet checkpoint..."
if [ ! -d "tfgridnet_for_urgent24" ]; then
  echo "Initializing git-lfs..."
  git lfs install

  echo "Downloading TF-GridNet checkpoint: wyz/tfgridnet_for_urgent24..."
  git clone https://huggingface.co/wyz/tfgridnet_for_urgent24

  echo "Done downloading TF-GridNet checkpoint."
else
  echo "TF-GridNet checkpoint already exists. Skipping."
fi

echo ""
echo "Done."