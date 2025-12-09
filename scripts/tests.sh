#!/bin/bash

pytest --cov=turtles --cov-report=term-missing --cov-config=.coveragerc -p no:warnings
