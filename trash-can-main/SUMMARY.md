**TrashCan 1.0: An Instance-Segmentation Labeled Dataset of Trash Observations** is a dataset for instance segmentation, semantic segmentation, and object detection tasks. It is used in the waste recycling, marine, and robotics industries. 

The dataset consists of 14424 images with 54285 labeled objects belonging to 22 different classes including *rov*, *trash unknown instance*, *trash plastic*, and other: *trash metal*, *trash etc*, *animal fish*, *plant*, *trash wood*, *animal eel*, *animal etc*, *trash fabric*, *animal crab*, *trash branch*, *trash fishing gear*, *animal shells*, *animal starfish*, *trash paper*, *trash rubber*, *trash wreckage*, *trash rope*, *trash tarp*, and *trash net*.

Images in the TrashCan 1.0 dataset have pixel-level instance segmentation and bounding box annotations. Due to the nature of the instance segmentation task, it can be automatically transformed into a semantic segmentation task (only one mask for every class). There are 129 (1% of the total) unlabeled images (i.e. without annotations). There are 4 splits in the dataset: *instance train* (6065 images), *materials train* (6008 images), *materials val* (1204 images), and *instance val* (1147 images). Additionally, every image marked with its ***video id*** tag. The dataset was released in 2020 by the University of Minnesota Twin Cities, USA.

Here are the visualized examples for the classes:

[Dataset classes](https://github.com/dataset-ninja/trash-can/raw/main/visualizations/classes_preview.webm)
