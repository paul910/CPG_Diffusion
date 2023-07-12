import configparser
from diffusion import Diffusion


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')

    Diffusion(config).start()


if __name__ == '__main__':
    main()
