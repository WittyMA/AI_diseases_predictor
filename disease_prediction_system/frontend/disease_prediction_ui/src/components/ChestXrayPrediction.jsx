import { useState, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Label } from '@/components/ui/label'
import { Loader2, Monitor, Upload, X, AlertTriangle, CheckCircle, Info, Image as ImageIcon } from 'lucide-react'
import { useToast } from '@/hooks/use-toast'

const ChestXrayPrediction = () => {
  const { toast } = useToast()
  const fileInputRef = useRef(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [selectedImage, setSelectedImage] = useState(null)
  const [imagePreview, setImagePreview] = useState(null)
  const [modelType, setModelType] = useState('pneumonia')

  const handleImageUpload = (e) => {
    const file = e.target.files[0]
    if (file) {
      if (file.type.startsWith('image/')) {
        setSelectedImage(file)
        
        // Create preview
        const reader = new FileReader()
        reader.onload = (e) => {
          setImagePreview(e.target.result)
        }
        reader.readAsDataURL(file)
        
        setResult(null) // Clear previous results
      } else {
        toast({
          title: "Invalid File",
          description: "Please select a valid image file.",
          variant: "destructive",
        })
      }
    }
  }

  const removeImage = () => {
    setSelectedImage(null)
    setImagePreview(null)
    setResult(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!selectedImage) {
      toast({
        title: "No Image Selected",
        description: "Please upload a chest X-ray image first.",
        variant: "destructive",
      })
      return
    }

    setLoading(true)
    setResult(null)

    try {
      // Convert image to base64
      const reader = new FileReader()
      reader.onload = async (e) => {
        try {
          const base64Image = e.target.result
          
          const response = await fetch('/api/predict/chest_xray', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              image: base64Image,
              model_type: modelType
            }),
          })

          if (!response.ok) {
            throw new Error('Prediction failed')
          }

          const data = await response.json()
          setResult(data)
          
          toast({
            title: "Analysis Complete",
            description: "Your chest X-ray analysis is ready.",
          })
        } catch (error) {
          toast({
            title: "Error",
            description: error.message || "Failed to analyze image. Please try again.",
            variant: "destructive",
          })
        } finally {
          setLoading(false)
        }
      }
      reader.readAsDataURL(selectedImage)
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to process image. Please try again.",
        variant: "destructive",
      })
      setLoading(false)
    }
  }

  const getPredictionColor = (prediction) => {
    if (modelType === 'pneumonia') {
      return prediction === 'PNEUMONIA' ? 'bg-red-500' : 'bg-green-500'
    } else {
      switch (prediction) {
        case 'COVID': return 'bg-red-500'
        case 'VIRAL_PNEUMONIA': return 'bg-orange-500'
        case 'NORMAL': return 'bg-green-500'
        default: return 'bg-gray-500'
      }
    }
  }

  const getPredictionIcon = (prediction) => {
    if (prediction === 'NORMAL') {
      return <CheckCircle className="mr-2 h-5 w-5" />
    } else {
      return <AlertTriangle className="mr-2 h-5 w-5" />
    }
  }

  return (
    <div className="max-w-6xl mx-auto space-y-8">
      {/* Header */}
      <div className="text-center space-y-4">
        <div className="flex items-center justify-center space-x-2">
          <Monitor className="h-8 w-8 text-blue-500" />/>
          <h1 className="text-3xl font-bold text-gray-900">Chest X-Ray Analysis</h1>
        </div>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Upload a chest X-ray image for AI-powered analysis to detect pneumonia or COVID-19.
        </p>
      </div>

      {/* Information Alert */}
      <Alert>
        <Info className="h-4 w-4" />
        <AlertDescription>
          This AI tool is for educational and research purposes only. It should not be used as a substitute 
          for professional medical diagnosis. Always consult with qualified healthcare professionals.
        </AlertDescription>
      </Alert>

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Upload Section */}
        <Card>
          <CardHeader>
            <CardTitle>Upload Chest X-Ray</CardTitle>
            <CardDescription>
              Select an analysis model and upload a clear chest X-ray image for AI analysis.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Model Selection */}
            <div className="space-y-2">
              <Label htmlFor="model-select">Analysis Model</Label>
              <Select value={modelType} onValueChange={setModelType}>
                <SelectTrigger>
                  <SelectValue placeholder="Select analysis model" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="pneumonia">Pneumonia Detection</SelectItem>
                  <SelectItem value="covid">COVID-19 Detection</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-sm text-gray-500">
                {modelType === 'pneumonia' 
                  ? 'Detects pneumonia vs normal chest X-rays'
                  : 'Detects COVID-19, viral pneumonia, and normal chest X-rays'
                }
              </p>
            </div>

            {/* Image Upload */}
            <div className="space-y-4">
              <Label>Chest X-Ray Image</Label>
              
              {!imagePreview ? (
                <div 
                  className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-gray-400 transition-colors cursor-pointer"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                  <p className="text-lg font-medium text-gray-900 mb-2">Upload X-Ray Image</p>
                  <p className="text-sm text-gray-500 mb-4">
                    Click to browse or drag and drop your chest X-ray image
                  </p>
                  <p className="text-xs text-gray-400">
                    Supported formats: JPG, PNG, JPEG (Max size: 10MB)
                  </p>
                </div>
              ) : (
                <div className="relative">
                  <img 
                    src={imagePreview} 
                    alt="Uploaded X-ray" 
                    className="w-full h-64 object-contain bg-gray-50 rounded-lg border"
                  />
                  <Button
                    variant="destructive"
                    size="icon"
                    className="absolute top-2 right-2"
                    onClick={removeImage}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              )}
              
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
              />
            </div>

            {/* Submit Button */}
            <Button 
              onClick={handleSubmit}
              className="w-full" 
              disabled={loading || !selectedImage}
            >
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Analyzing Image...
                </>
              ) : (
                <>
                  <ImageIcon className="mr-2 h-4 w-4" />
                  Analyze X-Ray
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Results Section */}
        <Card>
          <CardHeader>
            <CardTitle>Analysis Results</CardTitle>
            <CardDescription>
              AI-powered analysis results for the uploaded chest X-ray image.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {result ? (
              <div className="space-y-6">
                {/* Main Prediction */}
                <div className="text-center space-y-4">
                  <div className={`inline-flex items-center px-4 py-2 rounded-full text-white ${getPredictionColor(result.prediction)}`}>
                    {getPredictionIcon(result.prediction)}
                    <span className="font-semibold">{result.prediction}</span>
                  </div>
                  <p className="text-lg text-gray-900">
                    Model Prediction: <span className="font-bold">{result.prediction}</span>
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
                    {Object.entries(result.probabilities).map(([condition, probability]) => {
                      const isHighest = condition === result.prediction
                      return (
                        <div 
                          key={condition}
                          className={`flex justify-between items-center p-3 rounded-lg ${
                            isHighest ? 'bg-blue-50 border border-blue-200' : 'bg-gray-50'
                          }`}
                        >
                          <span className={`text-sm font-medium ${
                            isHighest ? 'text-blue-800' : 'text-gray-700'
                          }`}>
                            {condition}
                          </span>
                          <Badge 
                            variant="secondary" 
                            className={isHighest ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-700'}
                          >
                            {(probability * 100).toFixed(1)}%
                          </Badge>
                        </div>
                      )
                    })}
                  </div>
                </div>

                {/* Model Information */}
                <Alert>
                  <Info className="h-4 w-4" />
                  <AlertDescription>
                    Analysis performed using {modelType === 'pneumonia' ? 'Pneumonia Detection' : 'COVID-19 Detection'} model. 
                    This AI model has been trained on thousands of chest X-ray images for accurate pattern recognition.
                  </AlertDescription>
                </Alert>

                {/* Recommendations */}
                <Alert className={result.prediction === 'NORMAL' ? 'border-green-200 bg-green-50' : 'border-orange-200 bg-orange-50'}>
                  <AlertTriangle className="h-4 w-4" />
                  <AlertDescription>
                    {result.prediction === 'NORMAL' ? (
                      "The analysis suggests normal chest X-ray findings. Continue regular health monitoring."
                    ) : (
                      "The analysis suggests potential abnormalities. Please consult with a radiologist or healthcare provider for professional interpretation and further evaluation."
                    )}
                  </AlertDescription>
                </Alert>
              </div>
            ) : (
              <div className="text-center py-12 text-gray-500">
                  <Monitor className="mx-auto h-12 w-12 mb-4 opacity-50" />
                <p>Upload a chest X-ray image and click "Analyze X-Ray" to see results here.</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default ChestXrayPrediction

