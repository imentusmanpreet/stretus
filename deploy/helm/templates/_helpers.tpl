{{- define "python-ai.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "python-ai.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "python-ai.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "python-ai.labels" -}}
helm.sh/chart: {{ include "python-ai.chart" . }}
{{ include "python-ai.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: stretus-platform
app.kubernetes.io/component: python-ai
{{- end }}

{{- define "python-ai.selectorLabels" -}}
app.kubernetes.io/name: {{ include "python-ai.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "python-ai.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "python-ai.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{- define "python-ai.apiImage" -}}
{{- $global := index .Values "global" | default dict -}}
{{- $registry := index $global "imageRegistry" | default "" -}}
{{- $repository := .Values.api.image.repository -}}
{{- $tag := .Values.api.image.tag | default .Chart.AppVersion | default "latest" -}}
{{- if $registry }}
{{- printf "%s/%s:%s" $registry $repository $tag -}}
{{- else }}
{{- printf "%s:%s" $repository $tag -}}
{{- end }}
{{- end }}

{{- define "python-ai.marketDataGrpcTarget" -}}
{{- $svc := .Values.marketData.grpc.serviceName | default "marketdata-ingestion" -}}
{{- $port := .Values.marketData.grpc.port | default 50057 -}}
{{- $ns := .Values.marketData.grpc.namespace | default .Release.Namespace -}}
{{- printf "%s.%s.svc.cluster.local:%v" $svc $ns $port -}}
{{- end }}

{{/*
  Injected on the Deployment (not via .Values.env) so ArgoCD env overrides
  cannot strip gRPC settings. Matches BFF: marketdata-ingestion.<ns>.svc.cluster.local:50057
*/}}
{{- define "python-ai.marketDataEnv" -}}
{{- if .Values.marketData.grpc.enabled }}
- name: MARKET_DATA_FETCH_TRANSPORT
  value: {{ .Values.marketData.fetchTransport | default "grpc" | quote }}
- name: MARKET_DATA_GRPC_TARGET
  value: {{ include "python-ai.marketDataGrpcTarget" . | quote }}
- name: MARKET_DATA_GRPC_TIMEOUT_SECONDS
  value: {{ .Values.marketData.grpc.timeoutSeconds | default 120 | quote }}
- name: MARKET_DATA_GRPC_SECURE
  value: {{ .Values.marketData.grpc.secure | default false | quote }}
{{- end }}
{{- if .Values.marketData.httpFallback.historicalDataUrl }}
- name: HISTORICAL_DATA_URL
  value: {{ .Values.marketData.httpFallback.historicalDataUrl | quote }}
{{- end }}
- name: HISTORICAL_DATA_TIMEOUT_SECONDS
  value: {{ .Values.marketData.httpFallback.timeoutSeconds | default 120 | quote }}
{{- end }}

{{- define "python-ai.quantImage" -}}
{{- $global := index .Values "global" | default dict -}}
{{- $registry := index $global "imageRegistry" | default "" -}}
{{- $repository := .Values.quant.image.repository -}}
{{- $tag := .Values.quant.image.tag | default .Chart.AppVersion | default "latest" -}}
{{- if $registry }}
{{- printf "%s/%s:%s" $registry $repository $tag -}}
{{- else }}
{{- printf "%s:%s" $repository $tag -}}
{{- end }}
{{- end }}
