#!/bin/sh

echo "Script executed from: ${PWD}"

BASEDIR=$(dirname $0)
echo "Script location: ${BASEDIR}"
echo "\n\n"
cd .. && python -m master-thesis.genericExpandedClients.Mass_expanded_client 1 2 3