# DPIM

This repository host [code](./code), [samples and metrics](./results) for the [paper](https://doi.org/10.1007/978-3-031-30047-9_6) `Diverse Paraphrasing with Insertion Models for Few-Shot Intent Detection`. It proposes a pipeline for automatic and controllable paraphrase generation.

# License

Our code is licensed under dual [MIT](https://opensource.org/licenses/MIT) and [Apache v2](https://www.apache.org/licenses/LICENSE-2.0) licenses, at your choice. Some submodules may have more restrictive licenses, see the [Aknowledgments](./README.md#aknowledgments) section.

# References

You can link to this repo or/and cite our paper:
```bib
@InProceedings{10.1007/978-3-031-30047-9_6,
    author="Chevasson, Rapha{\"e}l
    and Laclau, Charlotte
    and Gravier, Christophe",
    editor="Cr{\'e}milleux, Bruno
    and Hess, Sibylle
    and Nijssen, Siegfried",
    title="Diverse Paraphrasing with Insertion Models for Few-Shot Intent Detection",
    booktitle="Advances in Intelligent Data Analysis XXI",
    year="2023",
    publisher="Springer Nature Switzerland",
    address="Cham",
    pages="65--76",
    isbn="978-3-031-30047-9"
}
```

# Inquiries

Feel free to contact us via github [issues](../../issues) or private message.

# Aknowledgments

Our work is based upon:
- [EWISER](https://github.com/SapienzaNLP/ewiser) ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)) for the synonym extraction,
- [POINTER](https://github.com/dreasysnail/POINTER) ([MIT](https://opensource.org/licenses/MIT)) for the controlable generation,
- [ProtAugment](https://github.com/tdopierre/ProtAugment) ([Apache v2](https://www.apache.org/licenses/LICENSE-2.0)) for fine-grained few-shot classification, to evaluate on the data augmentation task.

Kudo to them!
