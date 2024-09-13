## Pre-infection cerebral cortex structure predicts murine polymicrobial sepsis outcome

### We use high resolution structural brain MRIs and a murine polymicrobial sepsis LD50 model to demonstrate that pre-infection variability in brain structure can reliably predict infection outcome. Specifically, mice fated to survive exhibited greater cortical volume and thickness compared to those who succumbed. Our work reveals a readily measurable trait that can predict a murine LD50 polymicrobial sepsis outcome

### Methods

To predict a murine LD50 polymicrobial sepsis, we process our MRI images using [DeepBrainIPP](https://www.frontiersin.org/articles/10.3389/fbinf.2022.865443/full ). First, we update DeepBrainIPP model with ourdataset to perform skull Stripping. Then we perform images registration and morphological analysis. Finally, we build predictive models with measured morphologies. 

![MRI Reconstruction](data/pic.jpg?raw=true "Mouse Survival")

### Predictive modeling
We shared the complete data on morphological measurements as well as a python scripts (see "scripts folder") with step by step description on how models were built. The details can be found in our manuscripts (under review).

### Contact
Please feel free to contact (shahinur.alam0424@gmail.com, rmgallant5@gmail.com) if you have questions
