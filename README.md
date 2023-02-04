# Split_Manufacturing_One_Mask

This github repository is for the paper at DATE'23 - Split Additive Manufacturing for Printed Neuromorphic Circuits

cite as
```
Split Additive Manufacturing for Printed Neurmorphic Circuits
Zhao, H.; Hefenbrock, M.; Beigl, M.; Tahoori, M.
2023 Design, Automation & Test in Europe Conference & Exhibition (DATE), April 17-19, 2023, IEEE.
```

# Usage of the code
~~~
$ sh traincommand.sh
~~~

# Additional code

In addition to the pNNs described in the paper, we also implement the other two possible structure for split manufacturing.

1. all the tasks are using the same architecture, so that all the layers can be split manufactured. (which is the same as in the paper)

2. only hidden layers and output layers are having the same architecture, whereas the input layers are not shared. In this case, the input layers of each task (pNN) are different across all tasks and are trained fully independently.

3. only hidden layers are having the same architecture and shared among all tasks. Both input and output layers are tasks-specific and independent from other networks.

These three types are called full, semi, and hidden respectively.

