# Semi-Supervised-Intent-Classification

Intent Classification is the task of correctly labeling a natural language utterance from a predetermined set of intents. The task involves labeling the Engine-Design Dataset from Design Engineering Lab (DELP) @Purdue University to infer the decision-making processes by individuals in an engineering design team. The dataset contains the "body" column which has 7691 text message instances exchanged by the sender and the recipient. For the purpose of Semi-Supervised Intent Classification, different intents were identified after careful brainstorming to answer different questions to better understand the interactions and dialogue flow between the design agents in a design task.

For first intent, we want to understand the direction of information flow?
Option 1. Asking Information
Option 2. Providing Information:
Option 3. None of the above:

For the second intent, we want to understand which type of variables/information are the team members in a design team discussing?
Option 1. Talking about design parameters
       1.a. Dependencies between design parameters
       1.b. Exploration of design parameter values
Option 2. Talking about the objectives
       2.a. Tradeoff between objectives
       2.b. Monitoring objective values
Option 3. Talking about both design parameters and objectives
       3.a. Effects of design parameters on objectives
       3.b. Selected design parameter values for objective(s)
Option 4. None of the above

For the purpose of Semi-Supervised Intent Classification, 372 data instances were manually labeled with a total of 7319 unlabeled instances. Two different techniques of Semi-Supervised Intent Classification were employed for the intent prediction:
1. Usage of GAN + BERT technique
2. Usage of Self-Training using a deep learning model on top of Data Augmentation 

To compare both the methods, below is a table describing different performance metrics:


