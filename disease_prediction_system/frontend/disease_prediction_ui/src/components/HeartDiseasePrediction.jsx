import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Loader2, Heart, AlertTriangle, CheckCircle, Info } from 'lucide-react'
import { useToast } from '@/hooks/use-toast'

const HeartDiseasePrediction = () => {
  const { toast } = useToast()
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [formData, setFormData] = useState({
    age: '',
    sex: '',
    cp: '',
    trestbps: '',
    chol: '',
    fbs: '',
    restecg: '',
    thalach: '',
    exang: '',
    oldpeak: '',
    slope: '',
    ca: '',
    thal: ''
  })

  const handleInputChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const handleSelectChange = (name, value) => {
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
        if (value === '') {
          throw new Error(`Please fill in all fields`)
        }
        numericData[key] = parseFloat(value)
        if (isNaN(numericData[key])) {
          throw new Error(`Invalid value for ${key}`)
        }
      }

      const response = await fetch('/api/predict/heart_disease', {
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
        description: "Your heart disease risk assessment is ready.",
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
          <Heart className="h-8 w-8 text-pink-500" />
          <h1 className="text-3xl font-bold text-gray-900">Heart Disease Risk Prediction</h1>
        </div>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Enter your cardiovascular health metrics to assess your heart disease risk using our AI model.
        </p>
      </div>

      {/* Information Alert */}
      <Alert>
        <Info className="h-4 w-4" />
        <AlertDescription>
          This tool is for educational purposes only and should not replace professional medical advice. 
          Please consult with a cardiologist for proper cardiac evaluation.
        </AlertDescription>
      </Alert>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Input Form */}
        <Card>
          <CardHeader>
            <CardTitle>Cardiac Health Information</CardTitle>
            <CardDescription>
              Please provide your cardiovascular health metrics for accurate assessment.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Age */}
              <div className="space-y-2">
                <Label htmlFor="age">Age (years)</Label>
                <Input
                  id="age"
                  name="age"
                  type="number"
                  min="1"
                  max="120"
                  value={formData.age}
                  onChange={handleInputChange}
                  required
                  placeholder="Enter your age"
                />
              </div>

              {/* Sex */}
              <div className="space-y-2">
                <Label htmlFor="sex">Sex</Label>
                <Select value={formData.sex} onValueChange={(value) => handleSelectChange('sex', value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select sex" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1">Male</SelectItem>
                    <SelectItem value="0">Female</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Chest Pain Type */}
              <div className="space-y-2">
                <Label htmlFor="cp">Chest Pain Type</Label>
                <Select value={formData.cp} onValueChange={(value) => handleSelectChange('cp', value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select chest pain type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="0">Typical Angina</SelectItem>
                    <SelectItem value="1">Atypical Angina</SelectItem>
                    <SelectItem value="2">Non-Anginal Pain</SelectItem>
                    <SelectItem value="3">Asymptomatic</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Resting Blood Pressure */}
              <div className="space-y-2">
                <Label htmlFor="trestbps">Resting Blood Pressure (mmHg)</Label>
                <Input
                  id="trestbps"
                  name="trestbps"
                  type="number"
                  min="50"
                  max="250"
                  value={formData.trestbps}
                  onChange={handleInputChange}
                  required
                  placeholder="e.g., 120"
                />
              </div>

              {/* Cholesterol */}
              <div className="space-y-2">
                <Label htmlFor="chol">Serum Cholesterol (mg/dl)</Label>
                <Input
                  id="chol"
                  name="chol"
                  type="number"
                  min="100"
                  max="600"
                  value={formData.chol}
                  onChange={handleInputChange}
                  required
                  placeholder="e.g., 200"
                />
              </div>

              {/* Fasting Blood Sugar */}
              <div className="space-y-2">
                  <Label htmlFor="fbs">Fasting Blood Sugar &gt; 120 mg/dl</Label>
                <Select value={formData.fbs} onValueChange={(value) => handleSelectChange('fbs', value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select option" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1">Yes</SelectItem>
                    <SelectItem value="0">No</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Resting ECG */}
              <div className="space-y-2">
                <Label htmlFor="restecg">Resting ECG Results</Label>
                <Select value={formData.restecg} onValueChange={(value) => handleSelectChange('restecg', value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select ECG result" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="0">Normal</SelectItem>
                    <SelectItem value="1">ST-T Wave Abnormality</SelectItem>
                    <SelectItem value="2">Left Ventricular Hypertrophy</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Maximum Heart Rate */}
              <div className="space-y-2">
                <Label htmlFor="thalach">Maximum Heart Rate Achieved</Label>
                <Input
                  id="thalach"
                  name="thalach"
                  type="number"
                  min="60"
                  max="220"
                  value={formData.thalach}
                  onChange={handleInputChange}
                  required
                  placeholder="e.g., 150"
                />
              </div>

              {/* Exercise Induced Angina */}
              <div className="space-y-2">
                <Label htmlFor="exang">Exercise Induced Angina</Label>
                <Select value={formData.exang} onValueChange={(value) => handleSelectChange('exang', value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select option" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1">Yes</SelectItem>
                    <SelectItem value="0">No</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* ST Depression */}
              <div className="space-y-2">
                <Label htmlFor="oldpeak">ST Depression Induced by Exercise</Label>
                <Input
                  id="oldpeak"
                  name="oldpeak"
                  type="number"
                  min="0"
                  max="10"
                  step="0.1"
                  value={formData.oldpeak}
                  onChange={handleInputChange}
                  required
                  placeholder="e.g., 1.0"
                />
              </div>

              {/* Slope */}
              <div className="space-y-2">
                <Label htmlFor="slope">Slope of Peak Exercise ST Segment</Label>
                <Select value={formData.slope} onValueChange={(value) => handleSelectChange('slope', value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select slope" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="0">Upsloping</SelectItem>
                    <SelectItem value="1">Flat</SelectItem>
                    <SelectItem value="2">Downsloping</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Number of Major Vessels */}
              <div className="space-y-2">
                <Label htmlFor="ca">Number of Major Vessels Colored by Fluoroscopy</Label>
                <Select value={formData.ca} onValueChange={(value) => handleSelectChange('ca', value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select number" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="0">0</SelectItem>
                    <SelectItem value="1">1</SelectItem>
                    <SelectItem value="2">2</SelectItem>
                    <SelectItem value="3">3</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Thalassemia */}
              <div className="space-y-2">
                <Label htmlFor="thal">Thalassemia</Label>
                <Select value={formData.thal} onValueChange={(value) => handleSelectChange('thal', value)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1">Normal</SelectItem>
                    <SelectItem value="2">Fixed Defect</SelectItem>
                    <SelectItem value="3">Reversible Defect</SelectItem>
                  </SelectContent>
                </Select>
              </div>

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
                  'Predict Heart Disease Risk'
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
              AI-powered analysis of your heart disease risk based on the provided data.
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
                    {result.prediction === 1 ? 'Heart Disease Detected' : 'No Heart Disease Detected'}
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
                      <span className="text-sm font-medium text-green-800">No Heart Disease</span>
                      <Badge variant="secondary" className="bg-green-100 text-green-800">
                        {(result.probability.no_heart_disease * 100).toFixed(1)}%
                      </Badge>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-red-50 rounded-lg">
                      <span className="text-sm font-medium text-red-800">Heart Disease</span>
                      <Badge variant="secondary" className="bg-red-100 text-red-800">
                        {(result.probability.heart_disease * 100).toFixed(1)}%
                      </Badge>
                    </div>
                  </div>
                </div>

                {/* Recommendations */}
                <Alert>
                  <Info className="h-4 w-4" />
                  <AlertDescription>
                    {result.risk_level === 'High' ? (
                      "Consider consulting with a cardiologist for comprehensive cardiac evaluation and lifestyle modifications."
                    ) : (
                      "Maintain heart-healthy habits including regular exercise, balanced diet, and stress management."
                    )}
                  </AlertDescription>
                </Alert>
              </div>
            ) : (
              <div className="text-center py-12 text-gray-500">
                <Heart className="mx-auto h-12 w-12 mb-4 opacity-50" />
                <p>Fill in the form and click "Predict Heart Disease Risk" to see your results here.</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default HeartDiseasePrediction

