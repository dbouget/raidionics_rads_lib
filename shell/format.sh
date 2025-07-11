#!/bin/bash
isort --sl raidionicsrads
black --line-length 120 raidionicsrads
flake8 --max-line-length 120 --ignore "E203" raidionicsrads