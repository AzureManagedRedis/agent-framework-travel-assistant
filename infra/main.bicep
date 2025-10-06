targetScope = 'subscription'

@minLength(1)
@maxLength(64)
@description('Name of the environment that can be used as part of naming resource convention')
param environmentName string

@minLength(1)
@description('Primary location for all resources')
param location string

@description('Service name as defined in azure.yaml')
param serviceName string = 'web'

@secure()
@description('Tavily API Key')
param tavilyApiKey string

@secure()
@description('Mem0 API Key')
param mem0ApiKey string

// this tag tells azd which environment to use. The 'expirationfunction' name refers to the app in the azure.yaml file
var tags = {
  'azd-env-name': environmentName
}

// Create a new resource group
resource resourceGroup 'Microsoft.Resources/resourceGroups@2023-07-01' = {
  name: 'rg-${environmentName}'
  location: location
  tags: tags
}

// Deploy the resources in the resources.bicep file these resources are in a seprarate module because the scope of the resources is the resource group, not the subscription.
module resources './resources.bicep' = {
  name: 'resources'
  params: {
    location: location
    tags: tags
    serviceName: serviceName
    tavilyApiKey: tavilyApiKey
    mem0ApiKey: mem0ApiKey
  }
  scope : resourceGroup
}

output AZURE_CONTAINER_REGISTRY_ENDPOINT string = resources.outputs.AZURE_CONTAINER_REGISTRY_ENDPOINT
