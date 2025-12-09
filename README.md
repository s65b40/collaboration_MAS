
# Collaboration Strategies for Multi-Agent Systems

This repository contains the code for the ACL 2025 paper:

**Beyond Frameworks: Unpacking Collaboration Strategies in Multi-Agent Systems**

---

## Quick Start

### Data Preparation

In our study, we consider two task scenarios based on the AMBIFC and MIMIC-III datasets.  
This repository provides code for the AMBIFC dataset.

### Steps

1. Pre-process the AMBIFC dataset using:
   ```bash
   fact_code/filter_claim_data.py
    ```

2. The implementations of the nine collaboration strategies introduced in our paper are located in:

   ```bash
   fact_code/Gx_Px_Ix_Cx.py
   ```
    Details about the collaboration strategies can be found in our paper.
---

## Repository Overview

* `agent.py`: Defines and initializes agents.
* `api.py`: Handles calls to large language models (LLMs).
* `calculation_token.py`: Computes the token cost for a given LLM output.

---

## Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{wang-etal-2025-beyond,
    title = "Beyond Frameworks: Unpacking Collaboration Strategies in Multi-Agent Systems",
    author = "Wang, Haochun and Zhao, Sendong and Wang, Jingbo and Qiang, Zewen and Qin, Bing and Liu, Ting",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1037/",
    doi = "10.18653/v1/2025.acl-long.1037",
    pages = "21361--21375",
}
```

---

## Contact

If you have any questions about the paper or the code, feel free to contact: `hcwang@ir.hit.edu.cn`.

```
```
