# Calvera
In this project we will develop a Multi-Armed Bandits library in python that specializes on contextual neural bandit methods and combinatorial bandits.

![logo](./logo/calvera21.png)

## Architecture Overview
- Network Architecture (pytorch lightning module)
- Exploration Strategy (different Bandits)
- Vector Feedback (also combinatorial)
- Training / Optimization
For more details see ARCHITECTURE.md.

## MVP
### Baseline Algorithms
- [x] LinUCB
- [x] Linear Thompson Sampling

### Neural Algorithms
- [x] NeuralUCB
- [x] NeuralLinear
- [ ] NeuralTS
- [ ] Bootstrap
- [ ] Combinatorial Bandits

### Development Goals:
- The library should be extendable for implementing further methods and models.
- The trained model should be able to work stand-alone for inference
- The library is built on top of pytorch lightning => each model will be a `lightning.Module`.

### Evaluation Datasets
- [x] MNIST
- [x] Statlog
- [x] Covertype
- [x] Wheel Bandit
- [ ] RAG

### Further Optional Directions
- [ ] Norm-POEM style on logged bandit data
- [ ] LoRA style updates

[Link to Agreement](https://docs.google.com/document/d/1qs0hDGVd5MHe6PK5uL_GVNjiIePBJscbNkjGotF9-Uk/edit?tab=t.0])
