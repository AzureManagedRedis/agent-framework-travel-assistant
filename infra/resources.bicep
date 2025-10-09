@minLength(1)
@description('Primary location for all resources')
param location string = resourceGroup().location

@description('Tags to be applied to all resources')
param tags object = {}

@description('Service name as defined in azure.yaml')
param serviceName string = 'web'

@secure()
@description('Tavily API Key')
param tavilyApiKey string

@secure()
@description('Mem0 API Key')
param mem0ApiKey string

// Non-secure configuration params
param travelAgentModel string = 'demo-gpt-4o'
param mem0Model string = 'gpt-4o-mini'
param mem0EmbeddingModelDims int = 1536
param maxToolIterations int = 8
param maxChatHistorySize int = 10
param maxSearchResults int = 5
param share string = 'false'

param serverName string = '0.0.0.0'

param serverPort string = '7860'

param azureOpenAIApiVersion string = '2024-12-01-preview'

param azureOpenAIModelVersion string = '2024-08-06'

// Deployment name for the embeddings model used by Mem0
param mem0EmbeddingDeploymentName string = 'demo-emb-3-small'
// Optional: version for the embeddings model (many embeddings do not require a version)
param azureOpenAIEmbeddingModelVersion string = ''

var resourceToken = uniqueString(resourceGroup().id)

// Build embeddings model object conditionally including version only when provided
var embeddingModel = empty(azureOpenAIEmbeddingModelVersion) ? {
  format: 'OpenAI'
  name: 'text-embedding-3-small'
} : {
  format: 'OpenAI'
  name: 'text-embedding-3-small'
  version: azureOpenAIEmbeddingModelVersion
}

var resourceNames = {
  logAnalyticsWorkspaceName : 'log-${resourceToken}'
  containerAppName: 'ca-${resourceToken}'
  containerAppEnvironmentName: 'cae-${resourceToken}'
  azureOpenAIAccountName: 'oai-${resourceToken}'
  redisName: 'redis-${resourceToken}'
  redisAccessPolicyName: take('cachecontributor${resourceToken}', 24)
  containerRegistryName: 'acr${resourceToken}' // re-added for image pulls
  pullIdentityName: 'mi-pull-${resourceToken}' // user-assigned identity for ACR pulls
  keyVaultName: 'kv-${resourceToken}'
}


resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2025-02-01' = {
  name: resourceNames.logAnalyticsWorkspaceName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
  tags: tags
}

// Re-added ACR so container app system identity can pull images
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: resourceNames.containerRegistryName
  location: location
  sku: { name: 'Basic' }
  tags: tags
}

// User-assigned identity dedicated to pulling from ACR
resource pullIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: resourceNames.pullIdentityName
  location: location
  tags: tags
}

// Key Vault for secrets (RBAC-enabled)
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: resourceNames.keyVaultName
  location: location
  properties: {
    tenantId: tenant().tenantId
    enableRbacAuthorization: true
    sku: {
      family: 'A'
      name: 'standard'
    }
    publicNetworkAccess: 'Enabled'
  }
  tags: tags
}

// Store Tavily API key in Key Vault as 'tavily-api-key'
resource tavilySecret 'Microsoft.KeyVault/vaults/secrets@2019-09-01' = {
  name: 'tavily-api-key'
  parent: keyVault
  properties: {
    value: tavilyApiKey
  }
}

// Store Mem0 API key in Key Vault as 'mem0-api-key'
resource mem0Secret 'Microsoft.KeyVault/vaults/secrets@2019-09-01' = {
  name: 'mem0-api-key'
  parent: keyVault
  properties: {
    value: mem0ApiKey
  }
}

// Store Redis URL in Key Vault as 'redis-url'
resource redisUrlSecret 'Microsoft.KeyVault/vaults/secrets@2019-09-01' = {
  name: 'redis-url'
  parent: keyVault
  properties: {
    // Construct URL using the database primary key and standard Redis Enterprise FQDN pattern
    value: format('rediss://default:{0}@{1}.{2}.redis.azure.net:10000', redisDatabase.listKeys().primaryKey, resourceNames.redisName, toLower(location))
  }
}

// Store Azure OpenAI API key in Key Vault as 'azure-openai-api-key'
resource azureOpenAISecret 'Microsoft.KeyVault/vaults/secrets@2019-09-01' = {
  name: 'azure-openai-api-key'
  parent: keyVault
  properties: {
    value: azureOpenAIAccount.listKeys().key1
  }
  dependsOn: [
    azureOpenAIAccount
  ]
}

// Grant pullIdentity permission to read secrets in Key Vault (Key Vault Secrets User)
resource kvSecretsUserRA 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, pullIdentity.id, subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6'))
  scope: keyVault
  properties: {
    principalId: pullIdentity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6')
  }
}

resource redisEnterprise 'Microsoft.Cache/redisEnterprise@2024-09-01-preview' = {
  name: resourceNames.redisName
  location: location
  tags: tags
  sku: {
    name: 'Balanced_B0'
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    minimumTlsVersion: '1.2'
    highAvailability: 'Enabled'
  }
}

resource redisDatabase 'Microsoft.Cache/redisEnterprise/databases@2024-09-01-preview' = {
  name: 'default'
  parent: redisEnterprise
  properties: {
    clientProtocol: 'Encrypted'
    port: 10000
    clusteringPolicy: 'EnterpriseCluster'
    evictionPolicy: 'NoEviction'
    modules: [
      {
        name: 'RediSearch'
      }
      {
        name: 'RedisJSON'
      }
    ]
    persistence: {
      aofEnabled: false
      rdbEnabled: false
    }
    deferUpgrade: 'NotDeferred'
    accessKeysAuthentication: 'Enabled'
  }
}

resource azureOpenAIAccount 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: resourceNames.azureOpenAIAccountName
  location: location
  sku: {
    name: 'S0'
  }
  kind: 'OpenAI'
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    customSubDomainName: toLower(take('openai${uniqueString(resourceGroup().id)}', 24))
    publicNetworkAccess: 'Enabled'
    restore: false
    disableLocalAuth: false // allow API key usage since we provide OPENAI_API_KEY
  }
  tags: tags
}

resource gpt4o 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: azureOpenAIAccount
  name: 'demo-gpt-4o'
  sku: {
    name: 'Standard'
    capacity: 100
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4o'
      version: azureOpenAIModelVersion
    }
    currentCapacity: 100
  }
  tags: tags
}

// Embeddings deployment for Mem0 (text-embedding-3-small)
resource embSmall 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = {
  parent: azureOpenAIAccount
  name: mem0EmbeddingDeploymentName
  sku: {
    name: 'Standard'
    capacity: 1
  }
  properties: {
    model: embeddingModel
    currentCapacity: 1
  }
  tags: tags
  dependsOn: [
    gpt4o
  ]
}

resource env 'Microsoft.App/managedEnvironments@2024-02-02-preview' = {
  name: resourceNames.containerAppEnvironmentName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
  tags: tags
}

resource app 'Microsoft.App/containerApps@2024-02-02-preview' = {
  name: resourceNames.containerAppName
  location: location
  identity: {
    // Keep system for other resource access (OpenAI, Redis) + add user-assigned for ACR pull
    type: 'SystemAssigned,UserAssigned'
    userAssignedIdentities: {
      '${pullIdentity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      ingress: {
        external: true
        targetPort: 7860
      }
      registries: [
        {
          server: containerRegistry.properties.loginServer
          identity: pullIdentity.id // use user-assigned identity to authenticate to ACR
        }
      ]
      secrets: [
        {
          name: 'azure-openai-api-key'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/azure-openai-api-key'
          identity: pullIdentity.id
        }
        {
          name: 'tavily-api-key'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/tavily-api-key'
          identity: pullIdentity.id
        }
        {
          name: 'mem0-api-key'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/mem0-api-key'
          identity: pullIdentity.id
        }
        {
          name: 'redis-url'
          keyVaultUrl: '${keyVault.properties.vaultUri}secrets/redis-url'
          identity: pullIdentity.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: resourceNames.containerAppName
          image: 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest' // placeholder; azd deploy will override
          env: [
            { name: 'OPENAI_API_KEY', secretRef: 'azure-openai-api-key' }
            { name: 'AZURE_OPENAI_API_KEY', secretRef: 'azure-openai-api-key' }
            { name: 'AZURE_OPENAI_ENDPOINT', value: azureOpenAIAccount.properties.endpoint }
            { name: 'OPENAI_BASE_URL', value: '${azureOpenAIAccount.properties.endpoint}openai/v1/' }
            { name: 'AZURE_OPENAI_BASE_URL', value: '${azureOpenAIAccount.properties.endpoint}openai/v1/' }
            { name: 'AZURE_OPENAI_API_VERSION', value: azureOpenAIApiVersion }
            { name: 'OPENAI_API_VERSION', value: azureOpenAIApiVersion }
            { name: 'TAVILY_API_KEY', secretRef: 'tavily-api-key' }
            { name: 'MEM0_API_KEY', secretRef: 'mem0-api-key' }
            { name: 'SERVER_NAME', value: serverName }
            { name: 'SERVER_PORT', value: serverPort }
            { name: 'TRAVEL_AGENT_MODEL', value: travelAgentModel }
            { name: 'MEM0_MODEL', value: mem0Model }
            { name: 'MEM0_EMBEDDING_MODEL', value: mem0EmbeddingDeploymentName }
            { name: 'MEM0_EMBEDDING_MODEL_DIMS', value: string(mem0EmbeddingModelDims) }
            { name: 'MAX_TOOL_ITERATIONS', value: string(maxToolIterations) }
            { name: 'MAX_CHAT_HISTORY_SIZE', value: string(maxChatHistorySize) }
            { name: 'MAX_SEARCH_RESULTS', value: string(maxSearchResults) }
            { name: 'SHARE', value: share }
            { name: 'REDIS_URL', secretRef: 'redis-url' }
          ]
          resources: {           
            cpu: json('0.5')
            memory: '1Gi'
          }
        }
      ]
    }
  }
  tags: {
    'azd-service-name': serviceName
  }
  dependsOn: [
    acrPullRA
    kvSecretsUserRA
    redisEnterprise
    tavilySecret
    mem0Secret
    redisUrlSecret
    azureOpenAISecret
    embSmall
  ]
}

// AcrPull role for user-assigned identity (replaces earlier system identity assignment)
resource acrPullRA 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(containerRegistry.id, 'AcrPull', pullIdentity.name)
  scope: containerRegistry
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions','7f951dda-4ed3-4680-a7ca-43fe172d538d')
    principalId: pullIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Grant the container app's system identity access to Azure OpenAI (for future MI use)
resource azureOpenAIRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(azureOpenAIAccount.id, app.id, subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'a001fd3d-188f-4b5d-821b-7da978bf7442'))
  scope: azureOpenAIAccount
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'a001fd3d-188f-4b5d-821b-7da978bf7442')
    principalId: app.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// Assign Redis database access policy to the container app's system identity (for future Entra auth)
resource redisAccessPolicyAssignment 'Microsoft.Cache/redisEnterprise/databases/accessPolicyAssignments@2024-09-01-preview' = {
  name: resourceNames.redisAccessPolicyName
  parent: redisDatabase
  properties: {
    accessPolicyName: 'default'
    user: {
      objectId: app.identity.principalId
    }
  }
}

output WEB_URI string = app.properties.configuration.ingress.fqdn
output AZURE_LOG_ANALYTICS_WORKSPACE_NAME string = logAnalytics.name
output AZURE_LOG_ANALYTICS_WORKSPACE_ID string = logAnalytics.id
output AZURE_CONTAINER_APPS_ENVIRONMENT_NAME string = env.name
output AZURE_CONTAINER_APPS_ENVIRONMENT_ID string = env.id
output AZURE_CONTAINER_APPS_ENVIRONMENT_DEFAULT_DOMAIN string = env.properties.defaultDomain
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = containerRegistry.properties.loginServer
output KEY_VAULT_NAME string = keyVault.name
output KEY_VAULT_ID string = keyVault.id
