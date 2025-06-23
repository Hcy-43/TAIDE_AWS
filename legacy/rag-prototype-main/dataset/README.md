# public-datasets

## Info

This folder contains a list of traditional chinese/healthcare-related datasets from publicly available sources.

## Related Work

1. [藥學．要學：AI驅動的個人化精準學習](https://g0v.hackmd.io/8PYVkjW_RuSGFlIYJnGKoA?view): 本案目標在創建台灣醫療產業的藥劑共通LLM語言：收集在台灣醫療產業中的各種文件轉換成大語言模型(LLM)訓練用的藥學資料集。並且開源用此資料集微調過大語言模型(LLM)。

## Traditional Chinese Datasets

1. [【g0v 零時小學校】繁體中文 AI 開源實踐計畫](https://sch001.g0v.tw/dash/brd/2024TC-AI-OS-Grant/list)：【零時小學校】繁體中文 AI 開源實踐計畫，將針對「繁體中文文本採集」、「Benchmark 台灣觀點的評測」、「繁體中文特用型語言模型，以法律、醫療、教育為主但不限」、「繁體中文為主的開源模型」、「多語言考慮：台語、客語、原住民語」五大主題進行徵件。

2. [罕見疾病藥物資料庫暨線上通報系統/法規及相關下載](https://www.pharmaceutic.idv.tw/law_download.aspx)：

## Medical Datasets

1. [MedQuAD: Medical Question Answering Dataset](https://github.com/abachaa/MedQuAD): MedQuAD includes 47,457 medical question-answer pairs created from 12 NIH websites (e.g. cancer.gov, niddk.nih.gov, GARD, MedlinePlus Health Topics).
2. [MIMIC: Medical Information Mart for Intensive Care](https://mimic.mit.edu/): MIMIC (Medical Information Mart for Intensive Care) is a large, freely-available database comprising deidentified health-related data from patients who were admitted to the critical care units of the Beth Israel Deaconess Medical Center. This site contains information from the various versions of MIMIC released over the years:

   1. MIMIC-IV contains data from 2008-2019. The data was collected from Metavision bedside monitors.
   2. MIMIC-III contains data from 2001-2012. The data was collected from Metavision and CareVue bedside monitors.
   3. MIMIC-II contains data from 2001-2008. The data was collected from CareVue bedside monitors. MIMIC-II is no longer publicly available but the data can still be obtained from MIMIC-III by only including data from the CareVue monitors.
3. [4CE Dataset](https://portal.dbmi.hms.harvard.edu/projects/4ce-chatgpt-upload/): This international 4CE project aims to assess the capabilities of Large Language Models (LLMs) in analyzing de-identified clinical notes of obese patients with a history of COVID-19, with tasks including generating codes, translations, trial eligibility, guideline evaluation, SOAP notes, referrals, and medication summaries.
4. [MMLU: Measuring Massive Multitask Language Understanding]: MMLU (Massive Multitask Language Understanding) is a new benchmark designed to measure knowledge acquired during pretraining by evaluating models exclusively in zero-shot and few-shot settings.Relevent Tasks:
   1. college_medicine
   2. medical_genetics
   3. nutrition
   4. professional_medicine
   5. professional_psychology
   6. virology
5. [MedMentions: A UMLS Annotated Dataset](https://github.com/chanzuckerberg/MedMentions): MedMentions is a new manually annotated resource for the recognition of biomedical concepts. What distinguishes MedMentions from other annotated biomedical corpora is its size (over 4,000 abstracts and over 350,000 linked mentions), as well as the size of the concept ontology (over 3 million concepts from UMLS 2017) and its broad coverage of biomedical disciplines.
6. [PMC-OA: PubmedCentral OpenAcess](https://huggingface.co/datasets/axiong/pmc_oa): PMC-OA is a large-scale dataset that contains 1.65M image-text pairs. The figures and captions from PubMed Central, 2,478,267 available papers are covered and 12,211,907 figure-caption pairs are extracted.
7. [PMC-VQA: Visual Instruction Tuning for Medical Visual Question Answering](https://huggingface.co/datasets/xmcmic/PMC-VQA): PMC-VQA is a large-scale medical visual question-answering dataset that contains 227k VQA pairs of 149k images that cover various modalities or diseases. The question-answer pairs are generated from PMC-OA.
8. [NLP Mental Health Conversations](https://www.kaggle.com/datasets/thedevastator/nlp-mental-health-conversations): This dataset contains conversations between users and experienced psychologists related to mental health topics. Carefully collected and anonymized, the data can be used to further the development of Natural Language Processing (NLP) models which focus on providing mental health advice and guidance.