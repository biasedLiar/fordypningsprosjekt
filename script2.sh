#!/bin/sh

#cd /PycharmProjects/fordypningsprosjekt &&
#python ./genericClients/Mass_expanded_client.py

echo "Script executed from: ${PWD}"

BASEDIR=$(dirname $0)
echo "Script location: ${BASEDIR}"
echo "\n\n"
cd .. && python -m fordypningsprosjekt.genericExpandedClients.Mass_expanded_client 1 2 3