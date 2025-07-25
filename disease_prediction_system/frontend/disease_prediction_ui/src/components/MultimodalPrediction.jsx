import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
imimport { Brain, Info, Zap, Activity, Heart, Monitor, Construction } from 'lucide-react'

const MultimodalPrediction = () => {
  const [selectedModalities, setSelectedModalities] = useState([])

  const modalities = [
    {
      id: 'tabular',
      title: 'Tabular Data Analysis',
      description: 'Patient health metrics, lab results, and clinical measurements',
      icon: Activity,
      color: 'bg-blue-500',
      features: ['Blood tests', 'Vital signs', 'Medical history', 'Demographics']
    },
    {
      id: 'imaging',
      title: 'Medical Imaging',
      description: 'X-rays, CT scans, MRI, and other medical images',
      icon: Monitor,
      color: 'bg-green-500',
      features: ['Chest X-rays', 'CT scans', 'MRI images', 'Ultrasound']
    },
    {
      id: 'temporal',
      title: 'Time Series Data',
      description: 'ECG, EEG, continuous monitoring data over time',
      icon: Heart,
      color: 'bg-red-500',
      features: ['ECG signals', 'Heart rate variability', 'Blood pressure trends', 'Sleep patterns']
    }
  ]

  const advantages = [
    {
      title: 'Enhanced Accuracy',
      description: 'Combining multiple data types provides more comprehensive insights than single-modality analysis.',
      icon: Zap
    },
    {
      title: 'Robust Predictions',
      description: 'Multi-modal models are more resilient to missing or noisy data in individual modalities.',
      icon: Brain
    },
    {
      title: 'Holistic Assessment',
      description: 'Captures the full complexity of medical conditions through diverse data perspectives.',
      icon: Activity
    }
  ]

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center space-y-4">
        <div className="flex items-center justify-center space-x-2">
          <Brain className="h-8 w-8 text-purple-500" />
          <h1 className="text-3xl font-bold text-gray-900">Multi-Modal AI Prediction</h1>
        </div>
        <p className="text-gray-600 max-w-3xl mx-auto">
          Advanced AI system that combines multiple data types for comprehensive health assessment and disease prediction.
        </p>
      </div>

      {/* Development Notice */}
      <Alert className="border-orange-200 bg-orange-50">
        <Construction className="h-4 w-4" />
        <AlertDescription>
          <strong>Coming Soon:</strong> The multi-modal prediction system is currently under development. 
          This advanced feature will combine tabular data, medical imaging, and time-series analysis for enhanced accuracy.
        </AlertDescription>
      </Alert>

      {/* What is Multi-Modal AI */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Brain className="h-6 w-6 text-purple-500" />
            <span>What is Multi-Modal AI?</span>
          </CardTitle>
          <CardDescription>
            Understanding the power of combining multiple data types for medical prediction
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-gray-700 leading-relaxed">
            Multi-modal AI refers to artificial intelligence systems that can process and analyze multiple types of data 
            simultaneously. In healthcare, this means combining structured data (like lab results), medical images 
            (like X-rays), and time-series data (like ECG signals) to make more accurate and comprehensive predictions.
          </p>
          <p className="text-gray-700 leading-relaxed">
            By leveraging the strengths of different data modalities, these systems can capture patterns and relationships 
            that might be missed when analyzing each data type in isolation, leading to more robust and reliable medical insights.
          </p>
        </CardContent>
      </Card>

      {/* Supported Modalities */}
      <div className="space-y-6">
        <div className="text-center space-y-2">
          <h2 className="text-2xl font-bold text-gray-900">Supported Data Modalities</h2>
          <p className="text-gray-600">Our multi-modal system will integrate these different types of medical data</p>
        </div>
        
        <div className="grid md:grid-cols-3 gap-6">
          {modalities.map((modality) => {
            const Icon = modality.icon
            return (
              <Card key={modality.id} className="hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className={`w-12 h-12 ${modality.color} rounded-lg flex items-center justify-center mb-4`}>
                    <Icon className="h-6 w-6 text-white" />
                  </div>
                  <CardTitle className="text-xl">{modality.title}</CardTitle>
                  <CardDescription>{modality.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <h4 className="font-semibold text-sm text-gray-900">Key Features:</h4>
                    <ul className="space-y-1">
                      {modality.features.map((feature, index) => (
                        <li key={index} className="text-sm text-gray-600 flex items-center space-x-2">
                          <div className="w-1.5 h-1.5 bg-gray-400 rounded-full"></div>
                          <span>{feature}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </CardContent>
              </Card>
            )
          })}
        </div>
      </div>

      {/* Advantages */}
      <div className="space-y-6">
        <div className="text-center space-y-2">
          <h2 className="text-2xl font-bold text-gray-900">Advantages of Multi-Modal AI</h2>
          <p className="text-gray-600">Why combining multiple data types leads to better medical predictions</p>
        </div>
        
        <div className="grid md:grid-cols-3 gap-6">
          {advantages.map((advantage, index) => {
            const Icon = advantage.icon
            return (
              <Card key={index} className="text-center">
                <CardHeader>
                  <div className="mx-auto bg-gradient-to-r from-purple-600 to-indigo-600 w-12 h-12 rounded-lg flex items-center justify-center mb-4">
                    <Icon className="h-6 w-6 text-white" />
                  </div>
                  <CardTitle className="text-xl">{advantage.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-600">{advantage.description}</p>
                </CardContent>
              </Card>
            )
          })}
        </div>
      </div>

      {/* Technical Architecture */}
      <Card>
        <CardHeader>
          <CardTitle>Technical Architecture</CardTitle>
          <CardDescription>
            How our multi-modal system processes and combines different data types
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900">Data Processing Pipeline</h3>
              <div className="space-y-3">
                <div className="flex items-center space-x-3">
                  <Badge variant="outline" className="w-8 h-8 rounded-full flex items-center justify-center p-0">1</Badge>
                  <span className="text-sm text-gray-700">Data ingestion and preprocessing</span>
                </div>
                <div className="flex items-center space-x-3">
                  <Badge variant="outline" className="w-8 h-8 rounded-full flex items-center justify-center p-0">2</Badge>
                  <span className="text-sm text-gray-700">Modality-specific feature extraction</span>
                </div>
                <div className="flex items-center space-x-3">
                  <Badge variant="outline" className="w-8 h-8 rounded-full flex items-center justify-center p-0">3</Badge>
                  <span className="text-sm text-gray-700">Cross-modal attention mechanisms</span>
                </div>
                <div className="flex items-center space-x-3">
                  <Badge variant="outline" className="w-8 h-8 rounded-full flex items-center justify-center p-0">4</Badge>
                  <span className="text-sm text-gray-700">Unified prediction and confidence scoring</span>
                </div>
              </div>
            </div>
            
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-gray-900">Model Components</h3>
              <div className="space-y-2">
                <div className="p-3 bg-blue-50 rounded-lg">
                  <div className="font-medium text-blue-900">Tabular Encoder</div>
                  <div className="text-sm text-blue-700">Processes structured medical data</div>
                </div>
                <div className="p-3 bg-green-50 rounded-lg">
                  <div className="font-medium text-green-900">Vision Transformer</div>
                  <div className="text-sm text-green-700">Analyzes medical images</div>
                </div>
                <div className="p-3 bg-red-50 rounded-lg">
                  <div className="font-medium text-red-900">Temporal CNN</div>
                  <div className="text-sm text-red-700">Processes time-series signals</div>
                </div>
                <div className="p-3 bg-purple-50 rounded-lg">
                  <div className="font-medium text-purple-900">Fusion Network</div>
                  <div className="text-sm text-purple-700">Combines all modalities</div>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Current Status */}
      <Card>
        <CardHeader>
          <CardTitle>Development Status</CardTitle>
          <CardDescription>
            Current progress and upcoming features for the multi-modal prediction system
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
              <span className="font-medium text-green-900">Individual Disease Models</span>
              <Badge className="bg-green-100 text-green-800">✓ Complete</Badge>
            </div>
            <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
              <span className="font-medium text-green-900">Medical Image Analysis</span>
              <Badge className="bg-green-100 text-green-800">✓ Complete</Badge>
            </div>
            <div className="flex items-center justify-between p-3 bg-yellow-50 rounded-lg">
              <span className="font-medium text-yellow-900">Multi-Modal Fusion</span>
              <Badge className="bg-yellow-100 text-yellow-800">🔄 In Progress</Badge>
            </div>
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <span className="font-medium text-gray-900">Time-Series Integration</span>
              <Badge className="bg-gray-100 text-gray-800">⏳ Planned</Badge>
            </div>
          </div>
          
          <Alert>
            <Info className="h-4 w-4" />
            <AlertDescription>
              While the multi-modal system is under development, you can currently use our individual prediction models 
              for diabetes, heart disease, COVID-19 symptoms, and chest X-ray analysis. These models provide accurate 
              single-modality predictions and will be integrated into the unified multi-modal system.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>

      {/* Call to Action */}
      <div className="text-center space-y-4">
        <h2 className="text-2xl font-bold text-gray-900">Try Our Current Models</h2>
        <p className="text-gray-600 max-w-2xl mx-auto">
          While we're developing the multi-modal system, explore our individual AI models for specific health predictions.
        </p>
        <div className="flex flex-wrap justify-center gap-4">
          <Button asChild variant="outline">
            <a href="/diabetes">Diabetes Prediction</a>
          </Button>
          <Button asChild variant="outline">
            <a href="/heart-disease">Heart Disease</a>
          </Button>
          <Button asChild variant="outline">
            <a href="/covid-symptoms">COVID-19 Symptoms</a>
          </Button>
          <Button asChild variant="outline">
            <a href="/chest-xray">Chest X-Ray Analysis</a>
          </Button>
        </div>
      </div>
    </div>
  )
}

export default MultimodalPrediction

