#!/bin/bash

pip install -q paramiko

ORIGINAL_DIR=$(pwd)

cd `dirname $0`
WORKING_DIR=$(pwd)
DEV_DIR=${WORKING_DIR}'/../dev'

LOCAL_PRIVATE_KEY=${HOME}'/.ssh/MyKeyPair.pem'
NODE_PRIVATE_KEY=${HOME}'/.ssh/id_rsa'
GITHUB_HANDLER=${DEV_DIR}'/github_related.py'
PISCES_HANDLER=${DEV_DIR}'/pisces_related.py'
LAST_RESPONSE=${WORKING_DIR}'/last_response.json'
EC2_LAUNCH_RESULT=${WORKING_DIR}'/ec2_launch_result.json'

# Please change this to your forked repo for ease of your own debugging
GITHUB_REPO='git@github.com:SamuelGong/Pisces.git'
REPO_BRANCH='main'

case "$1" in
    install)
        python ${GITHUB_HANDLER} initialize ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE} ${NODE_PRIVATE_KEY} \
          ${GITHUB_REPO}
        python ${GITHUB_HANDLER} checkout ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE} ${GITHUB_REPO} \
          ${REPO_BRANCH}
        python ${PISCES_HANDLER} standalone ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE}
        ;;
    update)
        python ${GITHUB_HANDLER} pull ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE} ${GITHUB_REPO}
        ;;
    deploy_cluster)
        python ${PISCES_HANDLER} deploy_cluster ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE}
        ;;
    start_cluster)
        python ${PISCES_HANDLER} start_cluster ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE}
        ;;
    add_dependency)
        python ${PISCES_HANDLER} add_dependency ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE} "$2"
        ;;
    clean_memory)
        python ${PISCES_HANDLER} clean_memory ${EC2_LAUNCH_RESULT} \
          ${LOCAL_PRIVATE_KEY} ${LAST_RESPONSE}
        ;;
    *)
        echo "Unknown command!"
        ;;
esac

cd ${ORIGINAL_DIR}
