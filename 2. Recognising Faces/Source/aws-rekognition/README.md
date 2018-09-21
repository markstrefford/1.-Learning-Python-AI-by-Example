# Commands to set up AWS to use Rekognition

These commands will help you configure AWS for this module.  Note that you'll need to ensure that you set the relevant
regions for your AWS account (mine is `eu-west-1`).


### Install the AWS CLI
```bash
pip install awscli --upgrade
```

### Configure the AWS CLI
```bash
aws configure
```

### Rekognition Collection
```bash
aws rekognition create-collection --collection-id applied_ai_collection --region eu-west-1
```

### DynamoDB
```bash
aws dynamodb create-table --table-name applied_ai_collection \
--attribute-definitions AttributeName=RekognitionId,AttributeType=S \
--key-schema AttributeName=RekognitionId,KeyType=HASH \
--provisioned-throughput ReadCapacityUnits=1,WriteCapacityUnits=1 \
--region eu-west-1
```