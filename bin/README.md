## Modeling Scripts
These files perform the bulk of the modeling for this project.

`main.py`: extracts, transforms, and loads all data. Saves observations, mapper, and results to .pkl files.

`models.py`: functions which train estimators and return fitted estimators

`transformers.py`: functions for adding columns or transforming existing columns

`oregon_test.py`: a lightweight copy of `main.py` that extracts oregon data and transforms it. Does not train models, and does not produce a results file.