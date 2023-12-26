from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics, timing_metrics, cpu_usage_metrics, \
confusion_matrix_metrics, disk_usage_metrics
from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import Naive
from avalanche.core import SupervisedPlugin
# Load the MNIST dataset
from torchvision.datasets import MNIST


import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

class ReplayP(SupervisedPlugin):
    def __init__(self):
        super().__init__()
        self.count = 0


    def before_training_iteration(self, strategy, **kwargs):
        print(strategy.mb_x.size(0))
        for i in range(strategy.mb_x.size(0)):  # Iterate through the batch
            x, y, task = strategy.mb_x[i], strategy.mb_y[i], strategy.mb_task_id[i]
            image = x.cpu().detach()  # Image is already a 3D tensor
            label = y.cpu().detach().item()
            task_id = task.cpu().detach().item()
            # print(label, task_id)
            # findMNIST(task_id, label)
            pil_image = ToPILImage()(image)
            plt.imshow(pil_image)
            plt.title(f"Label: {label}; task id:{task_id}")

            # Save the figure to a file instead of displaying it
            plt.savefig(f"./sample/{self.count}_{label}_{task_id}.png")
            self.count += 1

            




benchmark = SplitMNIST(n_experiences=1, return_task_id=False)

# MODEL CREATION
model = SimpleMLP(num_classes=benchmark.n_classes)

# DEFINE THE EVALUATION PLUGIN and LOGGERS
# The evaluation plugin manages the metrics computation.
# It takes as argument a list of metrics, collectes their results and returns
# them to the strategy it is attached to.


loggers = []

# log to Tensorboard
loggers.append(TensorboardLogger())

# log to text file
loggers.append(TextLogger(open('loga.txt', 'a')))

# print to stdout
loggers.append(InteractiveLogger())

# # W&B logger - comment this if you don't have a W&B account
# loggers.append(WandBLogger(project_name="avalanche", run_name="test"))

eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    timing_metrics(epoch=True, epoch_running=True),
    cpu_usage_metrics(experience=True),
    forgetting_metrics(experience=True, stream=True),
    confusion_matrix_metrics(num_classes=benchmark.n_classes, save_image=True,
                             stream=True),
    disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loggers=loggers
)

# CREATE THE STRATEGY INSTANCE (NAIVE)
cl_strategy = Naive(
    model, SGD(model.parameters(), lr=0.001, momentum=0.9),
    CrossEntropyLoss(), train_mb_size=500, train_epochs=1, eval_mb_size=100,
    evaluator=eval_plugin, plugins=[ReplayP()])
    

# TRAINING LOOP
print('Starting experiment...')
results = []
accuracy_list = []
loss_list= []
i = 0
for experience in benchmark.train_stream:

    # The current Pytorch training set can be easily recovered through the
    # experience
    current_training_set = experience.dataset
    # ...as well as the task_label
    print('Task {}'.format(experience.task_label))
    print('This task contains', len(current_training_set), 'training examples')
    print("Current Classes: ", experience.classes_in_this_experience)
    
    # train returns a dictionary which contains all the metric values
    res = cl_strategy.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    # test also returns a dictionary which contains all the metric values
    results.append(cl_strategy.eval(benchmark.test_stream))
    
    acc = results[i]['Top1_Acc_Stream/eval_phase/test_stream/Task000']
    loss = results[i]['Loss_Stream/eval_phase/test_stream/Task000']
    accuracy_list.append(acc)
    loss_list.append(loss)
    i = i + 1
    

print(accuracy_list)
print(loss_list)

import matplotlib.pyplot as plt
# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(loss_list, label="Loss")
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(accuracy_list, label="Accuracy")
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# plt.show()
plt.savefig('result.png')
plt.close()  # Close the figure to free up memory
