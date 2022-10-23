<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GNU License][license-shield]][license-url]
<!-- [![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <!-- <a href="https://github.com/siemdejong/shg-strain-stress">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

<h3 align="center">Strain-Stress Curve Regression From SHG Images</h3>

  <p align="center">
    This project is a deep learning application to learn strain-stress curves from SHG skin tissue data.
    <br />
    <a href="https://siemdejong.github.io/shg-strain-stress"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <!-- <a href="https://github.com/siemdejong/shg-strain-stress">View Demo</a> -->
    ·
    <a href="https://github.com/siemdejong/shg-strain-stress/issues">Report Bug</a>
    ·
    <a href="https://github.com/siemdejong/shg-strain-stress/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->

The project aims to do a deep learning strain-stress curve regression, but uses second-harmonic generation (SHG) images to do so.
Using a large set of strain-stress and SHG of human skin tissue, a deep neural network is trained to predict the strain-stress curve of skin SHG images the network has never seen before.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

[![Python][Python]][Python-url]
[![PyTorch][PyTorch]][Pytorch-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This section includes instructions on setting up the project locally.

### Prerequisites
#### CUDA
The implementation allows use of CUDA enabled GPUs. 
Run `nvidia-smi` to see if there are CUDA enabled GPUs available.

#### Data
The current repository builds on three types of data:
1.  bitmap z-stacks
2.  target labels
3.  strain-stress curves with metadata

The tree below describes how the data should be organized.

```
TODO
```

### Installation

1.  Clone this repository.
    ```bash
    git clone https://github.com/siemdejong/shg-strain-stress.git shg-strain-stress
    ```
2.  Create a new conda environment and activate it.
    ```bash
    conda create -n <env_name>
    conda activate <env_name>
    ```
3.  Install dependencies from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
4.  Check if CUDA is available for the installed Pytorch distribution.
    In a Python shell, executre
    ```python
    import torch
    torch.cuda.is_available()
    ```
    If `false`, install Pytorch following its documentation.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

TODO: show examples of optuna optimization, model training, and inference.

_For more examples, please refer to the [documentation](https://siemdejong.github.io/shg-strain-stress)._

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Hyperparameter optimization with Optuna
- [ ] Model training with Pytorch
- [ ] Inference
- [ ] Explainable AI
- [ ] Documentation

See the [open issues](https://github.com/siemdejong/shg-strain-stress/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing
Contribute using the following steps.
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the GNU General Public License v3.0. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Siem de Jong - [linkedin.com/siemdejong](https://linkedin.com/siemdejong) - siem.dejong@hotmail.nl

Project Link: [https://github.com/siemdejong/shg-strain-stress](https://github.com/siemdejong/shg-strain-stress)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
<!-- ## Acknowledgments

* []()
* []()
* []()

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/siemdejong/shg-strain-stress.svg?style=for-the-badge
[contributors-url]: https://github.com/siemdejong/shg-strain-stress/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/siemdejong/shg-strain-stress.svg?style=for-the-badge
[forks-url]: https://github.com/siemdejong/shg-strain-stress/network/members
[stars-shield]: https://img.shields.io/github/stars/siemdejong/shg-strain-stress.svg?style=for-the-badge
[stars-url]: https://github.com/siemdejong/shg-strain-stress/stargazers
[issues-shield]: https://img.shields.io/github/issues/siemdejong/shg-strain-stress.svg?style=for-the-badge
[issues-url]: https://github.com/siemdejong/shg-strain-stress/issues
[license-shield]: https://img.shields.io/github/license/siemdejong/shg-strain-stress.svg?style=for-the-badge
[license-url]: https://github.com/siemdejong/shg-strain-stress/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/siemdejong
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[PyTorch]: https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[PyTorch-url]: https://pytorch.org
[Python]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://python.org
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com
