# Self-Adapting Language Models

[Paper](https://arxiv.org/abs/2506.10943), [Website](https://jyopari.github.io/posts/seal)

[Adam Zweiger](https://adamzweiger.github.io/),
[Jyothish Pari](https://jyopari.github.io),
[Han Guo](https://han-guo.info/),
[Ekin Akyürek](https://ekinakyurek.github.io/),
[Yoon Kim](https://people.csail.mit.edu/yoonkim/),
[Pulkit Agrawal](https://people.csail.mit.edu/pulkitag/)

MIT CSAIL

<img src="few-shot/assets/SEAL.png" alt="SEAL" width="400"/>

SEAL (**Se**lf-**A**dapting **L**LMs) is a framework for training language models via RL to generate self-edits (finetuning data and other update directives for themselves) in response to new inputs. 

We explore SEAL in two domains:
- [knowledge-incorporation](knowledge-incorporation): Incorporating new factual knowledge
- [few-shot](few-shot): Adapting to new tasks from few-shot examples

Both folders include code, data, and documentation.

## 🔧 Setup

### 1. Clone the repository

```bash
git clone https://github.com/Continual-Intelligence/SEAL.git
cd SEAL
```

### 2. Set up a virtual environment

Using **conda**:

```bash
conda create -n seal_env python=3.12
conda activate seal_env
```

Using **venv**:

```bash
python3.12 -m venv seal_env
source seal_env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

Create a `.env` file in the project root and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. SLURM users

Before running any shell scripts, make sure to update the SLURM directives at the top of each `.sh` file to match your system configuration.


## 📄 Citation

If you found this work useful, please cite:

```
@misc{zweiger2025selfadaptinglanguagemodels,
      title={Self-Adapting Language Models}, 
      author={Adam Zweiger and Jyothish Pari and Han Guo and Ekin Akyürek and Yoon Kim and Pulkit Agrawal},
      year={2025},
      eprint={2506.10943},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.10943}, 
}
```
