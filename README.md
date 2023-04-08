# **random AI notes**

## **Query Private Docs with GPT & LangChain** 
[April 2023]

[langchain/QueryMyDoc.ipynb](langchain/QueryMyDoc.ipynb)


[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rashlab/AI-Notes/blob/main/langchain/QueryMyDoc.ipynb)

### **About** 
This notebook summarizes a small exercise I conducted with LangChain, focusing on a significant challenge: leveraging LLMs to work with a corpus of private documents. There are numerous problems to tackle, and LangChain's example represents one of the first open-source attempts in this area. The notebook contains the LangChain code, my findings, and reflections. 

#


## the ***Transformer***  
[March 2023]

[Transformer/Transformer.ipynb](Transformer/Transformer.ipynb)

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rashlab/AI-Notes/blob/main/Transformer/Transformer.ipynb)

### **About** 
This notebook implements a mini-GPT model (tiny Shakespeare) I made while trying to understand Transformers, how and why they work so well. There is a long introductory section and plenty of notes added to the code, which is based entirely on Andrej Karpathy's [minGPT project](https://github.com/karpathy/minGPT). Thanks for that Andrej! 

* The model is trained on ~1 MB txt file of Shakespeare's writings, and after short training learns to generate new sonnets that, while nonsensical, resemble Shakespeare's style

* The model is using a simple character-level tokenizer, and the vocabulary size is relatively small - 65 unique characters: ```!$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz```

* The model is configured by default to run with 6 Transformer blocks, each with 6 masked self-attention heads, and the embedding size is 192

* The model has ~2.7 million learnable parameters, and on a single NVIDIA GeForce RTX 3090 it takes about 5 min to train. below is one sample of generated text:


```
In void of faith, and show thee men,
That together my daughters in heaven and thy growthry,
So labour'd lineaments are those my speaks,
Strikes tongues, for an ach ancient hests,
If you'll countenance him he all all hers.
Or will we defier it?

BRUTUS:
I'll none, but betimes our flatterer pleasure
Their love, and thine saltier are their halls
After hatches their song. Come, sometimes.

CORIOLANUS:
O groans!
A dog! thousand on this disland of hair,--
We'll be some of good spare out.

CORIOLANUS:
I must confess.
Ha
```


#


## Linear Transformation (with *PyTorch nn.Linear*)  
[March 2023]

### **About** 
This notebook is an elementary explainer on linear transformations in general, and specifically focuses on the PyTorch nn.Linear module. It also includes a basic exercise in **matplotlib** as a bonus

[nn.Linear/nn.Linear.ipynb](nn.Linear/nn.Linear.ipynb)

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rashlab/AI-Notes/blob/main/nn.Linear/nn.Linear.ipynb)

![basic linear transformation](filez/nn.Linear.png)



#




