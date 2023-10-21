# Summarizing-Radiology-Findings
# Project 
    Summarizing German Radiology Findings

# Requirments 
    torch<=1.7
    transformers<=4.3
    joeynmt
    pytorch_lightning<=1.5

# Data Format and Data Preparation
    path to example: "../../../../text_summarization.tsv"
    from data import ReportData
    data_loader = ReportData(tsv_path)
    train, valid, test = data_loader.get_middle_range_test_data()

# Model and Training Preparation
    from pointer_model.Seq2SeqDataset import SummaryData
    from pointer_model.EncoderDecoderModels import Bert2BertModel
    from joeynmt.helpers import load_config, set_seed

    configs = load_config(config_file)
    set_seed(seed=configs['train'].get('random_seed', 42))
    encoder_name = configs['model']['encoder']['name']
    decoder_name = configs['model']['decoder']['name']

    bert2bert = Bert2BertModel(config_file, pointer_ratio=1.0)
    bert2bert._load_pretrained_weights(encoder_name, decoder_name)

# Use train.py script and train a new model
    modify the model path in config file: configs/bert2bert_train.yaml
    run the train.py script: python train.py bert2bert_train.yaml

# Use generate.py script and generate summaries after training (test split)
    modify the model path in config file: configs/bert2bert_generate.yaml
    run the train.py script: python generate.py bert2bert_generate.yaml

# Generate only for one report contains general and findings sections (each section is a list of sentences)
    
    import os
    import torch

    checkpoint_path = os.path.join(model_dir, checkpoint_file)
    checkpoints = torch.load(checkpoint_path)
        bert2bert.load_state_dict(checkpoints['state_dict'])

    general = report[0][:2]
    findings = report[1]
    set_seed(seed=42)
    batch = {'src':['[CLS]'.join(general + findins)]}
    top_k_sents = {i:len(sent) for i, sent in enumerate(findings)} 
    sorted_topks = sorted(top_k_sents.items(), key=lambda item: item[1], reverse=True)[:4]
    indices = sorted([p[0] for p in sorted_topks])
    longest = ['[CLS]'.join([findings[i] for i in indices])]
    batch.update({'longest':longest})

    generations =  model._generate(batch, num_beams = 1, do_sample = False, num_sequences = 1, max_length = None, no_repeat_ngram_size = 5)
# Citation
    @inproceedings{liang-etal-2022-fine,
    title = "Fine-tuning {BERT} Models for Summarizing {G}erman Radiology Findings",
    author = "Liang, Siting  and
      Kades, Klaus  and
      Fink, Matthias  and
      Full, Peter  and
      Weber, Tim  and
      Kleesiek, Jens  and
      Strube, Michael  and
      Maier-Hein, Klaus",
    booktitle = "Proceedings of the 4th Clinical Natural Language Processing Workshop",
    month = jul,
    year = "2022",
    address = "Seattle, WA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.clinicalnlp-1.4",
    doi = "10.18653/v1/2022.clinicalnlp-1.4",
    pages = "30--40",}