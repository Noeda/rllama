#!/usr/bin/env python3

"""
This script uses the rllama API to generate tokens.

It does not print the tokens nicely.
"""

import requests

def main():
    url = 'http://127.0.0.1:8080/rllama/v1/inference'
    req = {
        'prompt': 'Hello world!',
        'max_seq_len': 1024,
        'max_new_tokens': 200,
        'no_token_sampling': False
    }
    res = requests.post(url, json=req, stream=True)
    for line in res.iter_lines():
        print(line.decode('utf-8'))


if __name__ == '__main__':
    main()
