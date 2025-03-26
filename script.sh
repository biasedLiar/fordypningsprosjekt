#!/bin/sh

#cd /PycharmProjects/fordypningsprosjekt &&
#python ./genericClients/Mass_testing_client.py

echo "Script executed from: ${PWD}"

BASEDIR=$(dirname $0)
echo "Script location: ${BASEDIR}"
echo "\n\n"
cd .. && python -m fordypningsprosjekt.genericClients.Mass_testing_client 1 2 3