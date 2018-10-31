# RatingNet : Learning driving habits using raw data

Instead of using a huge amount of data preprocessing and feature engineering, why not only feed the trip of a driver as a sequence of **state vectors** representing the trip. We can put as much data as we want in the state vector, such as latitude, longitude, velocity, ...

A similar approach can be used to do two things :

* Assess a *Safe Driving Rating* to a driver for a certain trip. This can be hard since we would need an accurate and unbiased annotated training set. However, this rating can be completely arbitrary, this may be hard to do in practice.
* Use the model for *Event Detection* for every trip point. This is also done using an annotated dataset, which we can automatically annotate using the current event detection methods we have (deterministic).

Both models are super simple. They consist of a multi-layer LSTM wich can be bidirectional or not. For the rating task, we only keep the output of the last neuron to compute the rating. We could also do a mean of all the ratings obtained at each time step... To see... The other only does a binary classification to detect events at each timestep.