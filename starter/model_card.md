# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a random forest classifier created by Trevor McCants on 4/19/2024 for Project 4 of the Udacity ML Ops nanodegree program.

## Intended Use

This model is intended to be used as an example model for learning about ML Ops.

## Training Data

This model was trained on census income data located [here](https://archive.ics.uci.edu/dataset/20/census+income). It contains information about people's demographics as well as if their income is above or below $50k.

## Evaluation Data

20% of the dataset mentioned above was not trained on. This data was used for testing the model's final performance.

## Metrics

Precision, recall, and fbeta score were used to evaluate model performance. This model specifically had a precision of 0.766, a recall of 0.617, and an fbeta score of 0.684.

## Ethical Considerations

Our model does not perform equally as well on all demographics. The model may be especially underfit on minorities since they represent a smaller percentage of our training data. One strategy used to quantify model bias was to evaluate the model's performance on slices of the data. These results can be found in "slice_output.txt"

## Caveats and Recommendations

This model was not well tuned. It is very likely that much higher performance could be achieved on this task. We recommend you make your own attempt at fitting this data instead of using this model.