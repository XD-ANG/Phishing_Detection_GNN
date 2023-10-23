### README

According to the report's statistics, the number of phishing attacks in the first half of 2023 reached 742.9 million times. It is evident that phishing attacks have become a significant threat in the cyberspace. We have proposed a novel phishing attack detection technique based on GNN. This detection is carried out solely based on URLs and is integrated into a heterogeneous graph.

#### Dataset

Use phishtank+virustotal to crawl my original dataset and then verify whether it is a phishing url or not, we collected the urls that have been published and verified in the recent months to ensure that we have a wide variety of types of phishing urls and benign urls.

#### Model Introduction

We used three common graph neural networks, GCN,GAT,GraphSAGE.

- **GCN（Graph Convolutional Network）：**GCN is a deep learning model for graph data, which achieves the learning of node embeddings by performing a convolution operation on the information of a node and its neighbors.
- **GAT（Graph Attention Network）：** GAT introduces an attention mechanism that allows it to dynamically learn node embeddings based on the importance between nodes.
- **GraphSAGE（Graph Sample and Aggregation）：**GraphSAGE achieves learning of node embedding by sampling neighboring nodes and aggregating their information.
