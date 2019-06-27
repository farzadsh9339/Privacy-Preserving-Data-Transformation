# Privacy-Preserving-Data-Transformation

Introduction:
With the advent of social networks and our daily moving online, vast amount of information are uploaded, collected, and shared daily online without the knowledge of data owners. The holder of the data, e.g., an internet service provider, usually wants to analyze them and make them public for research purposes, commercial, statistical, or other purposes. This obviously raises concerns about privacy issues for data contributors. We need to employ methods such that the data can be used successfully for the intended purpose, as not only are the data vulnerable to leakages, but also to malicious and adversary inference by other parties. Therefore,  privacy-protection methods should be employed that allow data collectors and owners to control the types of information that can be inferred from their data.

 Consider a scenario where mobile users upload their sensor readings to the cloud, which in turn trains a classifier that allows smartphones to identify their users from sensor readings in the background. This approach benefits from the huge storage and computation resources of the cloud. However, without proper processing the same data can be used to infer sensitive information about users, such as location and activities performed. This is especially dangerous when the information is considered private.

 Some features sent by the user to the cloud may produce a threat to privacy and could be used to extract sensitive and private personal information, such as age, gender, etc. A popular method named data perturbation introduces perturbation on the data provided by the user. The data have the same statistics as the true data and can be used for training the classifier without making the sensitive information available.

Random Projection is a method for data perturbation and also reducing the dimensionality of the data by more than 50%. Some papers treated this problem as a single classification task. They introduce a supervised version of Principal Component Analysis (PCA) called Discriminant Component Analysis (DCA) and the projection directions are affected by the within-class scatter matrix as in Linear Discriminant Analysis. RUCA can be considered as a mixture of DCA and MDR, and it can also be extended to privacy-sensitive classifications. Experimental results on Human Activity Recognition data set show that this methodology can provide better classification accuracies for the desired task while outperforming state-of-the-art privacy preserving data projection methods in terms of accuracies obtained from privacy-sensitive classifications.

Fundamentally, this problem can be viewed as a pair of classification tasks: one task is intended (for example, identification of the person’s gender) while the other task is undesirable (for example, identification of a person’s id). This task is called sensitive task In this context, we seek for an optimal data transformation that will maximally hurt the performance of any classifier for the undesired task without hurting the performance for the intended task. The other task which is called insensitive task is the one that we want to have a high classification accuracy. So our classifier we design should not hurt this task.
The methods are tested using the publicly available Human Activity Recognition data-set and the simulation results are reported in final section.

Methodology

The supposed data is described as follows: the user generates data in vectors that need to be uploaded to the cloud. Certain aspects of the data are not sensitive and it is the by the user to decide that the cloud provider can identify them. Some other aspects, however, are sensitive and their disclosure constitutes privacy leakage. In order to achieve privacy preservation, a system is needed so that it is possible to achieve the insensitive tasks while making it difficult to disclose privacy. In our scenario, the user does not upload the original data but rather a transformed or perturbed version which maximizes privacy protection. Our problem can be considered as a pair of classification tasks: a privacy insensitive (intended) task and a privacy sensitive (undesirable) task. The user is assumed to generate data as a vector sequence: 
Problem (A) (Privacy insensitive task)
Data set: {(x_1,t_1 ),(x_2,t_2 ),…,(x_N,t_N)}
Targets t_1 correspond to class labels related to insensitive information, e.g. gender. In general, there can be more than 2 classes: t_i=1,2,3,…,L
At the same time, given the same data sequence xi, we wish to make it difficult, if not impossible to solve the following classification problem:

Problem (B) (Privacy sensitive task)
Data set: {(x_1,s_1 ),(x_2,s_2 ),…,(x_P,s_P)}

Targets s_i are class labels related to sensitive information, for example, the ID’s of the persons that generate the data. Again there may be more than 2 classes: s_i= 1,2,3,…,P Our goal is to design data transformation so that any classifier would perform well on problem A but must perform poorly on problem B.
