

# Here is the source code of all agents

It includes **SWEagent, OpenHands, and ClaudeCode**. You can navigate into the corresponding folder and run our experiments according to its `README.md`.

# The experimental setup is as follows
| Experiment  | Description                                                 | loc   | knowledge | Test feedback |
| ----------- | ----------------------------------------------------------- | ----- | --------- | ------------- |
| 1 (default) | Only location information is provided                       | w.    | w.        | w\.o.         |
| 2           | Adds feedback for testing                                   | w.    | w.        | w.            |
| 3           | Adds feedback for testing, but without location information | w\.o. | w.        | w.            |
| 4           | No knowledge information provided                           | w.    | w\.o.     | w\.o.         |
| 5           | Blackbox                                                    | w\.o. | w.        | w\.o          |
