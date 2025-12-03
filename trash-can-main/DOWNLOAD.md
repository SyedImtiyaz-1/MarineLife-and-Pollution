Dataset **TrashCan 1.0** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://assets.supervisely.com/remote/eyJsaW5rIjogInMzOi8vc3VwZXJ2aXNlbHktZGF0YXNldHMvMzU5NV9UcmFzaENhbiAxLjAvdHJhc2hjYW4tMTAtRGF0YXNldE5pbmphLnRhciIsICJzaWciOiAiR2ZLcEtYZTFJSDU3T3lzMVV6dm9YLzhYeUFnbGVMSUpjZHlERU1wMXVYaz0ifQ==?response-content-disposition=attachment%3B%20filename%3D%22trashcan-10-DatasetNinja.tar%22)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='TrashCan 1.0', dst_dir='~/dataset-ninja/')
```
Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.

The data in original format can be [downloaded here](https://conservancy.umn.edu/bitstream/handle/11299/214865/dataset.zip?sequence=12&isAllowed=y).