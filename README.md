### README

According to the report's statistics, the number of phishing attacks in the first half of 2023 reached 742.9 million times. It is evident that phishing attacks have become a significant threat in the cyberspace. We have proposed a novel phishing attack detection technique based on GNN. This detection is carried out solely based on URLs and is integrated into a heterogeneous graph.

#### Dataset

Use phishtank+virustotal to crawl my original dataset and then verify whether it is a phishing url or not, we collected the urls that have been published and verified in the recent months to ensure that we have a wide variety of types of phishing urls and benign urls.

#### Model Introduction

We used three common graph neural networks, GCN,GAT,GraphSAGE.

- **GCN（Graph Convolutional Network）：**GCN is a deep learning model for graph data, which achieves the learning of node embeddings by performing a convolution operation on the information of a node and its neighbors.
- **GAT（Graph Attention Network）：** GAT introduces an attention mechanism that allows it to dynamically learn node embeddings based on the importance between nodes.
- **GraphSAGE（Graph Sample and Aggregation）：**GraphSAGE achieves learning of node embedding by sampling neighboring nodes and aggregating their information.

#### Code

The `code` directory contains scripts and implementations for constructing graphs using two different methods: `threshold` and `top-k`. These scripts are designed to create graphs using diverse methodologies, specifically employing the threshold-based and top-k approaches.

\- **Threshold Method:**When the similarity between two nodes (URLs) is greater than *θ*, an edge is added to connect the nodes.

\- **Top-K Method:**Each node is connected to the *k* nodes with the highest similarities.

#### requirements

To ensure smooth execution of the provided code, please refer to the `requirement.txt` file in the `code` directory. This file contains a list of dependencies and their versions required to run the scripts and execute the provided models.

It is recommended to set up a virtual environment and install the necessary dependencies using the following command:  pip install -r requirements.txt
