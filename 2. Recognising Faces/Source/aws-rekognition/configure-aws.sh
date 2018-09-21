#!/usr/bin/env bash

# Get region
DEFAULT_REGION = "eu-west-1"
read -e -p "Please enter your AWS region [$DEFAULT]:" REGION
REGION="${DEFAULT_REGION:-${DEFAULT}}"

# Get collection name
COLLECTION = "applied_ai_collection"

# Table name
TABLE_NAME="applied_ai_collection"

# Bucket name
BUCKET_NAME="applied_ai"

# Now install AWS CLI and configure
pip install awscli --upgrade && \
aws configure && \
aws rekognition create-collection --collection-id ${COLLECTION} --region ${REGION} && \
aws dynamodb create-table --table-name ${TABLE_NAME} \
--attribute-definitions AttributeName=RekognitionId,AttributeType=S \
--key-schema AttributeName=RekognitionId,KeyType=HASH \
--provisioned-throughput ReadCapacityUnits=1,WriteCapacityUnits=1 \
--region ${REGION}

# Create an AWS Bucket
aws s3 mb s3://${BUCKET_NAME} --region ${REGION}

# Create the trust policy
aws iam create-role --role-name LambdaRekognitionRole --assume-role-policy-document file://trust-policy.json

# Attach the access policy
aws iam put-role-policy --role-name LambdaRekognitionRole --policy-name LambdaPermissions --policy-document file://access-policy.json

