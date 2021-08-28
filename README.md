# Stochastic Physics-Informed Neural Networks (SPINN)

Stochastic differential equations (SDEs) are used to describe a wide variety of complex stochastic dynamical systems. Learning the hidden physics within SDEs is crucial for unraveling fundamental understanding of these systems’ stochastic and nonlinear behavior. We propose a flexible and scalable framework for training deep neural networks to learn constitutive equations that represent hidden physics within SDEs. The proposed stochastic physics-informed neural network framework (SPINN) relies on uncertainty propagation and moment-matching techniques along with state-of-the-art deep learning strategies. SPINN first propagates stochasticity through the known structure of the SDE (i.e., the known physics) to predict the time evolution of statistical moments of the stochastic states. SPINN learns (deep) neural network representations of the hidden physics by matching the predicted moments to those estimated from data. Recent advances in automatic differentiation and mini-batch gradient descent are leveraged to establish the unknown parameters of the neural networks. SPINN provides a promising new direction for systematically unraveling the hidden physics of multivariate stochastic dynamical systems with multiplicative noise.

# Help
Please direct all questions to jared.oleary@berkeley.edu




