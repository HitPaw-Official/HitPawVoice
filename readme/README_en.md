[简体中文](README_cn.md) | English

<div align="center">
  <img src="../docs/logo.png" alt="HitPawVoice Logo" width="100%"/>
</div>

<p align="center">
  <a href="../LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://github.com/HitPaw-Official"><img src="https://img.shields.io/badge/org-HitPaw--Official-orange" alt="Org"></a>
  <img src="https://img.shields.io/badge/python-3.8%2B-brightgreen" alt="Python">
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey" alt="Platform">
</p>

---

## Introduction

**HitPawVoice** is the official index repository for all audio and voice-related AI algorithms maintained by [HitPaw-Official](https://github.com/HitPaw-Official). This repository does **not** contain algorithm source code directly — it serves as a navigation hub with curated links, descriptions, and quick-start guides pointing to each standalone algorithm repository.

All algorithm repositories are independently versioned, documented, and deployable.

---

## Algorithm Directory

| Algorithm | Description | Repo Link | Paper |
|-----------|-------------|-----------|-------|
| **BeatDetection** | Rhythm and beat detection from audio signals | [HitPaw-Official/BeatDetection](https://github.com/HitPaw-Official/BeatDetection) | — |
| **SingingVoiceConversion** | High-quality any-to-any singing voice style transfer | [HitPaw-Official/SingingVoiceConversion](https://github.com/HitPaw-Official/SingingVoiceConversion) | — |

---

## Repository Structure (per algorithm)

Each algorithm repository under HitPaw-Official follows this layout:

```
AlgorithmName/
├── configs/          # Model and training configs
├── data/             # Dataset preparation scripts
├── deploy/           # Export, ONNX, and serving tools
├── docs/             # Extended documentation and assets
├── models/           # Model definitions
├── tools/            # Training / evaluation entry points
├── README.md
└── requirements.txt
```

---

## Quick Start

Visit the specific algorithm repo for environment setup. All repos share a common pattern:

```bash
# 1. Clone the target algorithm repo
git clone https://github.com/HitPaw-Official/<AlgorithmName>.git
cd <AlgorithmName>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download pretrained weights (see each repo's README)

# 4. Run inference — see the specific algorithm repo's README for the exact command
```

---

## Related Hubs

| Hub | Domain | Link |
|-----|--------|-------|
| HitPawImage | Image AI algorithms | [HitPaw-Official/HitPawImage](https://github.com/HitPaw-Official/HitPawImage) |
| HitPawVideo | Video AI algorithms | [HitPaw-Official/HitPawVideo](https://github.com/HitPaw-Official/HitPawVideo) |

---

## Contributing

We welcome contributions to any algorithm under HitPaw-Official. Please open issues or pull requests in the **specific algorithm repository** rather than this hub.

For organization-wide discussions, open an issue here.

---

## Support

- [Official Website](https://www.hitpaw.com/)
- [Developer Portal](https://developer.hitpaw.com/)
- [Get API Key](https://www.hitpaw.com/hitpaw-api.html)

---

## License

All repositories under HitPaw-Official are released under the [Apache 2.0 License](../LICENSE) unless otherwise noted in the individual repository.
