#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 14:07:25 2021

@author: gustavo
"""
cd Downloads/mpfa-d
sudo docker run --rm -d -v $PWD:/el --name mpfad gustavo_mpfa_d
sudo docker exec -it mpfad /bin/bash
cd /el
ipython3