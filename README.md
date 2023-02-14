Welcome to PBARLIKE's documentation!
====================================

[![made-with-python](https://img.shields.io/badge/Version-1.0.0-1f425f.svg)](https://www.python.org/)

Indirect searches for Dark Matter (DM) look for messengers such as gamma-rays, antiprotons and heavier 
anti-nuclei that can be produced in DM annihilation in the sky. With the release of 7-year data from
AMS-02 [^1], highly accurate antiproton measurements have become available. Traditional analysis with 
CR nuclei involves solving coupled differential equations to solve for CR propagation, and is hence
copmutationally very heavy. Global fits of antiproton data with other experiments
are crucial in determining status of different DM models. In the context of global fits, such CR propagation
with traditional simulators become prohibitively expensive. Thus the speed-up provided by emulators built from deep neural networks
are crucial for global fits.

**DarkRayNet**[^2] (DRN), a deep neural network provides significant speed up in predicting propagation of CR nuclei. 
**GAMBIT** [^3] is an open-source, global fitting framework developed for global fits in Beyond-the-Standard-Model Physics. The code 
**PBARLIKE** [^4] is an addition to this family of numerical codes, developed for performing convenient and computationally efficient analyses 
for DM searches with antiprotons. 

The code PBARLIKE obtains antiproton flux predictions from DRN, and calculates likelihoods using the recent AMS-02 data. The likelihood is
marginalized over the nuisance parameters from propagation and solar modulation. It also involves state-of-the-art
treatment of correlations in data. Most importantly, PBARLIKE includes the gambit_interface module that
allows access from within GAMBIT to fast AMS-02 antiproton likelihood calculation using DRN.

Citations
=========

Please cite arXiv:23xx.xxxxx and ..?

Documentation
=============



References:
-----------

[^1] AMS Collaboration, M. Aguilar et al., [The Alpha Magnetic Spectrometer (AMS) on the international space station: Part II — Results from the first seven years](https://www.sciencedirect.com/science/article/pii/S0370157320303434?via%3Dihub), Phys. Rept. 894 (2021) 1–116.

[^2] F. Kahlhoefer, M. Korsmeier, M. Kr ̈amer, S. Manconi, and K. Nippel, [Constraining dark matter annihilation with cosmic ray antiprotons using neural networks](https://iopscience.iop.org/article/10.1088/1475-7516/2021/12/037), JCAP 12 (2021), no. 12 037, [[2107.12395]](https://arxiv.org/abs/2107.12395).

[^3] GAMBIT Collaboration, P. Athron et al., [GAMBIT: The Global and Modular Beyond-the-Standard-Model Inference Tool](https://link.springer.com/article/10.1140/epjc/s10052-017-5321-8), Eur. Phys. J. C 77 (2017), no. 11 784, [[1705.07908]](https://arxiv.org/abs/1705.07908). [Addendum: Eur.Phys.J.C 78, 98 (2018)].

[^4] Add reference.

[^5] J. Heisig, M. Korsmeier, and M. W. Winkler, [Dark matter or correlated errors: Systematics of the AMS-02 antiproton excess](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043017), Phys. Rev. Res. 2 (2020), no. 4 043017, [[2005.04237]](https://arxiv.org/abs/2005.04237).

