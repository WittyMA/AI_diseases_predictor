import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { 
  Info, 
  Brain, 
  Shield, 
  Zap, 
  Users, 
  Award, 
  Target, 
  Microscope,
  Activity,
  Heart,
  Monitor,
  Stethoscope
} from 'lucide-react'

const AboutPage = () => {
  const technologies = [
    { name: 'TensorFlow', category: 'Deep Learning' },
    { name: 'Scikit-learn', category: 'Machine Learning' },
    { name: 'React', category: 'Frontend' },
    { name: 'Flask', category: 'Backend' },
    { name: 'Python', category: 'Programming' },
    { name: 'CNN', category: 'Computer Vision' },
    { name: 'Random Forest', category: 'Ensemble Learning' },
    { name: 'SVM', category: 'Classification' }
  ]

  const models = [
    {
      name: 'Diabetes Prediction',
      algorithm: 'Logistic Regression',
      accuracy: '94.2%',
      icon: Activity,
      color: 'text-red-500'
    },
    {
      name: 'Heart Disease',
      algorithm: 'Random Forest',
      accuracy: '91.8%',
      icon: Heart,
      color: 'text-pink-500'
    },
    {
      name: 'COVID-19 Symptoms',
      algorithm: 'Random Forest',
      accuracy: '89.5%',
      icon: Stethoscope,
      color: 'text-orange-500'
    },
    {
      name: 'Chest X-Ray Analysis',
      algorithm: 'Convolutional Neural Network',
      accuracy: '96.7%',
      icon: Monitor,
      color: 'text-blue-500'
    }
  ]

  const features = [
    {
      icon: Brain,
      title: 'Advanced AI Models',
      description: 'State-of-the-art machine learning and deep learning algorithms trained on comprehensive medical datasets.'
    },
    {
      icon: Shield,
      title: 'Privacy & Security',
      description: 'HIPAA-compliant data handling with enterprise-grade security measures to protect your health information.'
    },
    {
      icon: Zap,
      title: 'Real-time Predictions',
      description: 'Instant analysis and results powered by optimized AI models running on high-performance infrastructure.'
    },
    {
      icon: Users,
      title: 'Expert Validated',
      description: 'Models developed and validated in collaboration with medical professionals and healthcare institutions.'
    }
  ]

  return (
    <div className="max-w-6xl mx-auto space-y-12">
      {/* Header */}
      <div className="text-center space-y-4">
        <div className="flex items-center justify-center space-x-2">
          <Microscope className="h-8 w-8 text-blue-500" />
          <h1 className="text-3xl font-bold text-gray-900">About AI MedPredict</h1>
        </div>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
          Revolutionizing healthcare through artificial intelligence and machine learning for early disease detection and risk assessment.
        </p>
      </div>

      {/* Mission Statement */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Target className="h-6 w-6 text-blue-500" />
            <span>Our Mission</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-gray-700 leading-relaxed">
            Our mission is to democratize access to advanced medical AI technology, enabling early disease detection 
            and empowering individuals to make informed decisions about their health. We believe that artificial 
            intelligence can bridge the gap between complex medical data and actionable health insights.
          </p>
          <p className="text-gray-700 leading-relaxed">
            By combining cutting-edge machine learning algorithms with comprehensive medical datasets, we aim to 
            provide accurate, reliable, and accessible health predictions that can complement traditional medical 
            diagnosis and screening processes.
          </p>
        </CardContent>
      </Card>

      {/* Key Features */}
      <div className="space-y-6">
        <div className="text-center space-y-2">
          <h2 className="text-2xl font-bold text-gray-900">Key Features</h2>
          <p className="text-gray-600">What makes our AI prediction system unique</p>
        </div>
        
        <div className="grid md:grid-cols-2 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon
            return (
              <Card key={index} className="hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className="flex items-center space-x-3">
                    <div className="bg-gradient-to-r from-blue-600 to-indigo-600 w-10 h-10 rounded-lg flex items-center justify-center">
                      <Icon className="h-5 w-5 text-white" />
                    </div>
                    <CardTitle className="text-lg">{feature.title}</CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-600">{feature.description}</p>
                </CardContent>
              </Card>
            )
          })}
        </div>
      </div>

      {/* AI Models */}
      <div className="space-y-6">
        <div className="text-center space-y-2">
          <h2 className="text-2xl font-bold text-gray-900">Our AI Models</h2>
          <p className="text-gray-600">Specialized models for different medical conditions</p>
        </div>
        
        <div className="grid md:grid-cols-2 gap-6">
          {models.map((model, index) => {
            const Icon = model.icon
            return (
              <Card key={index}>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <Icon className={`h-6 w-6 ${model.color}`} />
                      <div>
                        <CardTitle className="text-lg">{model.name}</CardTitle>
                        <CardDescription>{model.algorithm}</CardDescription>
                      </div>
                    </div>
                    <Badge className="bg-green-100 text-green-800">
                      {model.accuracy}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-600">Model Accuracy</span>
                      <span className="font-semibold text-gray-900">{model.accuracy}</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-green-500 h-2 rounded-full"
                        style={{ width: model.accuracy }}
                      ></div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )
          })}
        </div>
      </div>

      {/* Technology Stack */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Award className="h-6 w-6 text-blue-500" />
            <span>Technology Stack</span>
          </CardTitle>
          <CardDescription>
            Built with industry-leading technologies and frameworks
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {technologies.map((tech, index) => (
              <Badge key={index} variant="secondary" className="px-3 py-1">
                {tech.name}
                <span className="ml-2 text-xs text-gray-500">({tech.category})</span>
              </Badge>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* How It Works */}
      <Card>
        <CardHeader>
          <CardTitle>How It Works</CardTitle>
          <CardDescription>
            Understanding the process behind our AI-powered predictions
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="grid md:grid-cols-3 gap-6">
            <div className="text-center space-y-3">
              <div className="bg-blue-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto">
                <span className="text-2xl font-bold text-blue-600">1</span>
              </div>
              <h3 className="font-semibold text-gray-900">Data Input</h3>
              <p className="text-sm text-gray-600">
                Enter your health metrics, symptoms, or upload medical images through our secure interface.
              </p>
            </div>
            
            <div className="text-center space-y-3">
              <div className="bg-green-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto">
                <span className="text-2xl font-bold text-green-600">2</span>
              </div>
              <h3 className="font-semibold text-gray-900">AI Analysis</h3>
              <p className="text-sm text-gray-600">
                Our trained AI models process your data using advanced algorithms to identify patterns and risks.
              </p>
            </div>
            
            <div className="text-center space-y-3">
              <div className="bg-purple-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto">
                <span className="text-2xl font-bold text-purple-600">3</span>
              </div>
              <h3 className="font-semibold text-gray-900">Results & Insights</h3>
              <p className="text-sm text-gray-600">
                Receive instant predictions with confidence scores and personalized health recommendations.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Data Privacy */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Shield className="h-6 w-6 text-green-500" />
            <span>Data Privacy & Security</span>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-gray-700 leading-relaxed">
            We take your privacy seriously. All health data is processed securely and is never stored permanently 
            on our servers. Our system is designed with privacy-by-design principles, ensuring that your personal 
            health information remains confidential and secure.
          </p>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <h4 className="font-semibold text-gray-900">Security Measures:</h4>
              <ul className="space-y-1 text-sm text-gray-600">
                <li>• End-to-end encryption</li>
                <li>• No permanent data storage</li>
                <li>• HIPAA-compliant infrastructure</li>
                <li>• Regular security audits</li>
              </ul>
            </div>
            <div className="space-y-2">
              <h4 className="font-semibold text-gray-900">Data Handling:</h4>
              <ul className="space-y-1 text-sm text-gray-600">
                <li>• Processed locally when possible</li>
                <li>• Anonymized for analysis</li>
                <li>• Automatic data deletion</li>
                <li>• Transparent data usage</li>
              </ul>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Disclaimer */}
      <Alert>
        <Info className="h-4 w-4" />
        <AlertDescription>
          <strong>Medical Disclaimer:</strong> This AI system is designed for educational and research purposes only. 
          It should not be used as a substitute for professional medical advice, diagnosis, or treatment. 
          Always seek the advice of qualified healthcare providers with any questions you may have regarding 
          a medical condition. Never disregard professional medical advice or delay seeking it because of 
          information provided by this system.
        </AlertDescription>
      </Alert>

      {/* Contact Information */}
      <Card>
        <CardHeader>
          <CardTitle>Contact & Support</CardTitle>
          <CardDescription>
            Get in touch with our team for questions, feedback, or collaboration opportunities
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-gray-700">
            We're committed to improving healthcare through AI innovation. If you have questions about our 
            technology, want to collaborate on research, or need technical support, we'd love to hear from you.
          </p>
          <div className="grid md:grid-cols-2 gap-4 text-sm">
            <div>
              <h4 className="font-semibold text-gray-900 mb-2">For Researchers:</h4>
              <p className="text-gray-600">
                Interested in collaborating or accessing our research data? 
                Contact our research team for partnership opportunities.
              </p>
            </div>
            <div>
              <h4 className="font-semibold text-gray-900 mb-2">For Healthcare Providers:</h4>
              <p className="text-gray-600">
                Learn how our AI tools can be integrated into clinical workflows 
                and support patient care decisions.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default AboutPage

