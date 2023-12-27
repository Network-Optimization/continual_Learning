# continual_Learning

Implementation of continuous learning related algorithms and many of my own work.

Including the 中科软件终.py, which uses multiple fusion algorithms and embeds an entity extraction model, selecting the most annotated dataset for the entity extraction model of the confidential system and providing their connections.

Continuous learning algorithms that integrate multiple strategies typically have better performance than single algorithms because multiple strategies provide diversity in the dataset. In a fixed dataset standard task, the better the diversity of the dataset, the more likely the model is to "consider the big picture" and learn more complete knowledge.

However, existing papers only modify different strategy algorithms or use combinations of multiple strategy algorithms, without in-depth quantitative analysis of how different strategy algorithms can enhance ACC. Fusion strategies are essentially a method that combines the computational results of multiple algorithms.

I have conducted preliminary work to analyze the performance of multi strategy fusion algorithms in improving the model. I have confirmed in CL.pdF that the allocation of strategies for different algorithms can affect the overall performance of the model. Does this mean that the combination of internal mechanisms of several algorithms can represent multiple internal mechanism methods (or alternatives)? I will continue to research and make existing algorithm methods public.
#

Other instructions waiting for supplementation.
