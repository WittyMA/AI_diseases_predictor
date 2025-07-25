import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Loader2, Activity, AlertTriangle, CheckCircle, Info } from 'lucide-react'
import { useToast } from '@/hooks/use-toast'

const DiabetesPrediction = () => {
  const { toast } = useToast()
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [formData, setFormData] = useState({
    Pregnancies: '',
    Glucose: '',
    BloodPressure: '',
    SkinThickness: '',
    Insulin: '',
    BMI: '',
    DiabetesPedigreeFunction: '',
    Age: ''
  })

  const handleInputChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setResult(null)

    try {
      // Convert form data to numbers
      const numericData = {}
      for (const [key, value] of Object.entries(formData)) {
        numericData[key] = parseFloat(value)
        if (isNaN(numericData[key])) {
          throw new Error(`Invalid value for ${key}`)
        }
      }

      const response = await fetch('/api/predict/diabetes', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(numericData),
      })

      if (!response.ok) {
        throw new Error('Prediction failed')
      }

      const data = await response.json()
      setResult(data)
      
      toast({
        title: "Prediction Complete",
        description: "Your diabetes risk assessment is ready.",
      })
    } catch (error) {
      toast({
        title: "Error",
        description: error.message || "Failed to get prediction. Please try again.",
        variant: "destructive",
      })
    } finally {
      setLoading(false)
    }
  }

  const formFields = [
    { name: 'Pregnancies', label: 'Number of Pregnancies', type: 'number', min: 0, max: 20, step: 1 },
    { name: 'Glucose', label: 'Glucose Level (mg/dL)', type: 'number', min: 0, max: 300, step: 1 },
    { name: 'BloodPressure', label: 'Blood Pressure (mmHg)', type: 'number', min: 0, max: 200, step: 1 },
    { name: 'SkinThickness', label: 'Skin Thickness (mm)', type: 'number', min: 0, max: 100, step: 1 },
    { name: 'Insulin', label: 'Insulin Level (μU/mL)', type: 'number', min: 0, max: 1000, step: 1 },
    { name: 'BMI', label: 'Body Mass Index', type: 'number', min: 0, max: 70, step: 0.1 },
    { name: 'DiabetesPedigreeFunction', label: 'Diabetes Pedigree Function', type: 'number', min: 0, max: 3, step: 0.001 },
    { name: 'Age', label: 'Age (years)', type: 'number', min: 1, max: 120, step: 1 }
  ]

  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'High': return 'bg-red-500'
      case 'Low': return 'bg-green-500'
      default: return 'bg-gray-500'
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center space-y-4">
        <div className="flex items-center justify-center space-x-2">
          <Activity className="h-8 w-8 text-red-500" />
          <h1 className="text-3xl font-bold text-gray-900">Diabetes Risk Prediction</h1>
        </div>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Enter your health metrics below to assess your diabetes risk using our AI-powered prediction model.
        </p>
      </div>

      {/* Information Alert */}
      <Alert>
        <Info className="h-4 w-4" />
        <AlertDescription>
          This tool is for educational purposes only and should not replace professional medical advice. 
          Please consult with a healthcare provider for proper medical evaluation.
        </AlertDescription>
      </Alert>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Input Form */}
        <Card>
          <CardHeader>
            <CardTitle>Patient Information</CardTitle>
            <CardDescription>
              Please fill in all the required health metrics for accurate prediction.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              {formFields.map((field) => (
                <div key={field.name} className="space-y-2">
                  <Label htmlFor={field.name}>{field.label}</Label>
                  <Input
                    id={field.name}
                    name={field.name}
                    type={field.type}
                    min={field.min}
                    max={field.max}
                    step={field.step}
                    value={formData[field.name]}
                    onChange={handleInputChange}
                    required
                    placeholder={`Enter ${field.label.toLowerCase()}`}
                  />
                </div>
              ))}
              <Button 
                type="submit" 
                className="w-full" 
                disabled={loading}
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  'Predict Diabetes Risk'
                )}
              </Button>
            </form>
          </CardContent>
        </Card>

        {/* Results */}
        <Card>
          <CardHeader>
            <CardTitle>Prediction Results</CardTitle>
            <CardDescription>
              AI-powered analysis of your diabetes risk based on the provided data.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {result ? (
              <div className="space-y-6">
                {/* Risk Level */}
                <div className="text-center space-y-4">
                  <div className={`inline-flex items-center px-4 py-2 rounded-full text-white ${getRiskColor(result.risk_level)}`}>
                    {result.risk_level === 'High' ? (
                      <AlertTriangle className="mr-2 h-5 w-5" />
                    ) : (
                      <CheckCircle className="mr-2 h-5 w-5" />
                    )}
                    <span className="font-semibold">{result.risk_level} Risk</span>
                  </div>
                  <p className="text-2xl font-bold text-gray-900">
                    {result.prediction === 1 ? 'Diabetes Detected' : 'No Diabetes Detected'}
                  </p>
                </div>

                {/* Confidence Score */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-gray-700">Confidence</span>
                    <span className="text-sm font-bold text-gray-900">
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${result.confidence * 100}%` }}
                    ></div>
                  </div>
                </div>

                {/* Probability Breakdown */}
                <div className="space-y-3">
                  <h4 className="font-semibold text-gray-900">Probability Breakdown</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between items-center p-3 bg-green-50 rounded-lg">
                      <span className="text-sm font-medium text-green-800">No Diabetes</span>
                      <Badge variant="secondary" className="bg-green-100 text-green-800">
                        {(result.probability.no_diabetes * 100).toFixed(1)}%
                      </Badge>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-red-50 rounded-lg">
                      <span className="text-sm font-medium text-red-800">Diabetes</span>
                      <Badge variant="secondary" className="bg-red-100 text-red-800">
                        {(result.probability.diabetes * 100).toFixed(1)}%
                      </Badge>
                    </div>
                  </div>
                </div>

                {/* Recommendations */}
                <Alert>
                  <Info className="h-4 w-4" />
                  <AlertDescription>
                    {result.risk_level === 'High' ? (
                      "Consider consulting with a healthcare provider for further evaluation and lifestyle recommendations."
                    ) : (
                      "Maintain a healthy lifestyle with regular exercise and balanced diet to keep your risk low."
                    )}
                  </AlertDescription>
                </Alert>
              </div>
            ) : (
              <div className="text-center py-12 text-gray-500">
                <Activity className="mx-auto h-12 w-12 mb-4 opacity-50" />
                <p>Fill in the form and click "Predict Diabetes Risk" to see your results here.</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default DiabetesPrediction

