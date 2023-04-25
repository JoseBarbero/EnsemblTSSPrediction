# EnsemblTSSPrediction

Recognizing transcription start sites is key to gene identification. Several approaches have been employed in related problems such as detecting translation initiation sites or promoters, many of the most recent ones based on machine learning. Deep learning methods have been proven to be exceptionally effective for this task, but their use in transcription start site identification has not yet been explored in depth. Also, the very few existing works do not compare their methods to support vector machines (SVMs), the most established technique in this area of study, nor provide the curated dataset used in the study. The reduced amount of published papers in this specific problem could be explained by this lack of datasets. Given that both support vector machines and deep neural networks have been applied in related problems with remarkable results, we compared their performance in transcription start site predictions, concluding that SVMs are computationally much slower, and deep learning methods, specially long short-term memory neural networks (LSTMs), are best suited to work with sequences than SVMs. For such a purpose, we used the reference human genome GRCh38. Additionally, we studied two different aspects related to data processing: the proper way to generate training examples and the imbalanced nature of the data. Furthermore, the generalization performance of the models studied was also tested using the mouse genome, where the LSTM neural network stood out from the rest of the algorithms. To sum up, this article provides an analysis of the best architecture choices in transcription start site identification, as well as a method to generate transcription start site datasets including negative instances on any species available in Ensembl. We found that deep learning methods are better suited than SVMs to solve this problem, being more efficient and better adapted to long sequences and large amounts of data. We also create a transcription start site (TSS) dataset large enough to be used in deep learning experiments.

## Dataset

Human (GRCh38) Dataset: https://doi.org/10.5281/zenodo.7147597

Mouse Dataset: https://doi.org/10.5281/zenodo.7679000

The steps to recreate the dataset can be found in the [curate_data](https://github.com/JoseBarbero/EnsemblTSSPrediction/tree/main/curate_data) folder.

## Citation
Barbero-Aparicio JA, Olivares-Gil A, Díez-Pastor JF, García-Osorio C. 2023. Deep learning and support vector machines for transcription start site identification. PeerJ Computer Science 9:e1340 https://doi.org/10.7717/peerj-cs.1340

### Bibtex
```
@article{barbero-aparicio2023,
  title = {Deep learning and support vector machines for transcription start site identification},
  author = {José A. Barbero-Aparicio and Alicia Olivares-Gil and José F. Díez-Pastor and César García-Osorio},
  editor = {Carlos Fernandez-Lozano},
  url = {https://doi.org/10.7717/peerj-cs.1340},
  doi = {10.7717/peerj-cs.1340},
  issn = {2376-5992},
  year = {2023},
  date = {2023-04-17},
  urldate = {2023-04-17},
  journal = {PeerJ Computer Science},
  volume = {9},
  issue = {e1340},
  abstract = {Recognizing transcription start sites is key to gene identification. Several approaches have been employed in related problems such as detecting translation initiation sites or promoters, many of the most recent ones based on machine learning. Deep learning methods have been proven to be exceptionally effective for this task, but their use in transcription start site identification has not yet been explored in depth. Also, the very few existing works do not compare their methods to support vector machines (SVMs), the most established technique in this area of study, nor provide the curated dataset used in the study. The reduced amount of published papers in this specific problem could be explained by this lack of datasets. Given that both support vector machines and deep neural networks have been applied in related problems with remarkable results, we compared their performance in transcription start site predictions, concluding that SVMs are computationally much slower, and deep learning methods, specially long short-term memory neural networks (LSTMs), are best suited to work with sequences than SVMs. For such a purpose, we used the reference human genome GRCh38. Additionally, we studied two different aspects related to data processing: the proper way to generate training examples and the imbalanced nature of the data. Furthermore, the generalization performance of the models studied was also tested using the mouse genome, where the LSTM neural network stood out from the rest of the algorithms. To sum up, this article provides an analysis of the best architecture choices in transcription start site identification, as well as a method to generate transcription start site datasets including negative instances on any species available in Ensembl. We found that deep learning methods are better suited than SVMs to solve this problem, being more efficient and better adapted to long sequences and large amounts of data. We also create a transcription start site (TSS) dataset large enough to be used in deep learning experiments},
  keywords = {bioinformatics, Convolutional neural network, Deep learning, Long short-term memory, Machine learning, Support vector machines, transcription start site},
  pubstate = {published},
  tppubtype = {article}
}
```
