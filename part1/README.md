Lab 1 - Task 1
===================


![Result](https://github.com/iloveyii/data-mining-1/blob/master/task2/result.png)

# Objective
In this part, you shall
1. Learn how to use Cloud Computing for your Data Mining tasks

# Google cloud
Google cloud provides two services for data mining, ie Data lab and Virtual private server. It is like a Jupyter like interface.
Google cloud provides a trial free services for learning purposes. 

## Virtual Machines
		Service: Compute > Compute engine

![Screenshot](https://github.com/iloveyii/data-mining-1/blob/master/task2/screenshot.png)


* To connect to virtual machine use the following command:
```bash
gcloud compute ssh --project project-id --zone zone-name instance-name
```
## Datalab

* Create an instance by using datalab command.
```bash
datalab create fis
```

* The command will ask for region. Choose the one that suits you the best.
		
![Screenshot](https://github.com/iloveyii/data-mining-1/blob/master/task1/images/dl-select-region.png)

* It will take some time to create the environment. Once ready it will guide you to start the service.

![Screenshot](https://github.com/iloveyii/data-mining-1/blob/master/part1/images/dl-change-port.png)

		
* If for any reason you disconnect the datalab image, you can reconnect using:

```python
datalab connect fis
```

* Now data lab is ready to work with, create a notebook and happy coding.
		
![Screenshot](https://github.com/iloveyii/data-mining-1/blob/master/part1/images/datalab_fis.png)
