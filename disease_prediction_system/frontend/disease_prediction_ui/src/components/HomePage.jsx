import { Link } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { 
  Activity, 
  Heart, 
  Stethoscope, 
  Monitor, 
  Brain, 
  Shield, 
  Zap, 
  Users,
  ArrowRight,
  CheckCircle
} from 'lucide-react'

const HomePage = () => {
  const predictionServices = [
    {
      title: 'Diabetes Prediction',
      description: 'Predict diabetes risk based on patient health metrics and lifestyle factors.',
      icon: Activity,
      path: '/diabetes',
      color: 'bg-red-500',
      features: ['Blood glucose analysis', 'BMI assessment', 'Family history']
    },
    {
      title: 'Heart Disease Prediction',
      description: 'Assess cardiovascular risk using comprehensive cardiac health indicators.',
      icon: Heart,
      path: '/heart-disease',
      color: 'bg-pink-500',
      features: ['ECG analysis', 'Blood pressure', 'Cholesterol levels']
    },
    {
      title: 'COVID-19 Symptoms',
      description: 'Evaluate COVID-19 infection probability based on symptoms and exposure.',
      icon: Stethoscope,
      path: '/covid-symptoms',
      color: 'bg-orange-500',
      features: ['Symptom analysis', 'Contact tracing', 'Risk assessment']
    },
    {
      title: 'Chest X-Ray Analysis',
      description: 'AI-powered analysis of chest X-rays for pneumonia and COVID-19 detection.',
      icon: Monitor,
      path: '/chest-xray',
      color: 'bg-blue-500',
      features: ['Deep learning CNN', 'Pneumonia detection', 'COVID-19 screening']
    },
    {
      title: 'Multi-Modal Prediction',
      description: 'Advanced AI combining multiple data types for comprehensive health assessment.',
      icon: Brain,
      path: '/multimodal',
      color: 'bg-purple-500',
      features: ['Combined analysis', 'Enhanced accuracy', 'Holistic assessment']
    }
  ]

  const systemFeatures = [
    {
      icon: Shield,
      title: 'HIPAA Compliant',
      description: 'Your health data is protected with enterprise-grade security'
    },
    {
      icon: Zap,
      title: 'Real-time Results',
      description: 'Get instant predictions powered by state-of-the-art AI models'
    },
    {
      icon: Users,
      title: 'Expert Validated',
      description: 'Models trained and validated by medical professionals'
    }
  ]

  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <section className="text-center space-y-6">
        <div className="space-y-4">
          <Badge variant="secondary" className="px-4 py-2 text-sm">
            🚀 Powered by Advanced AI & Machine Learning
          </Badge>
          <h1 className="text-4xl md:text-6xl font-bold text-gray-900 leading-tight">
            AI-Powered Disease
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-indigo-600">
              {' '}Prediction System
            </span>
          </h1>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
            Harness the power of artificial intelligence for early disease detection and health risk assessment. 
            Our multi-modal system combines tabular data analysis with medical imaging for comprehensive health insights.
          </p>
        </div>
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Button size="lg" className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700">
            <Link to="/diabetes" className="flex items-center space-x-2">
              <span>Start Prediction</span>
              <ArrowRight className="h-4 w-4" />
            </Link>
          </Button>
          <Button size="lg" variant="outline">
            <Link to="/about">Learn More</Link>
          </Button>
        </div>
      </section>

      {/* Features Section */}
      <section className="space-y-8">
        <div className="text-center space-y-4">
          <h2 className="text-3xl font-bold text-gray-900">Why Choose Our AI System?</h2>
          <p className="text-gray-600 max-w-2xl mx-auto">
            Built with cutting-edge technology and validated by medical experts
          </p>
        </div>
        <div className="grid md:grid-cols-3 gap-6">
          {systemFeatures.map((feature, index) => {
            const Icon = feature.icon
            return (
              <Card key={index} className="text-center hover:shadow-lg transition-shadow">
                <CardHeader>
                  <div className="mx-auto bg-gradient-to-r from-blue-600 to-indigo-600 w-12 h-12 rounded-lg flex items-center justify-center mb-4">
                    <Icon className="h-6 w-6 text-white" />
                  </div>
                  <CardTitle className="text-xl">{feature.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-gray-600">{feature.description}</p>
                </CardContent>
              </Card>
            )
          })}
        </div>
      </section>

      {/* Prediction Services */}
      <section className="space-y-8">
        <div className="text-center space-y-4">
          <h2 className="text-3xl font-bold text-gray-900">Available Prediction Services</h2>
          <p className="text-gray-600 max-w-2xl mx-auto">
            Choose from our comprehensive suite of AI-powered health prediction tools
          </p>
        </div>
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {predictionServices.map((service, index) => {
            const Icon = service.icon
            return (
              <Card key={index} className="hover:shadow-xl transition-all duration-300 hover:-translate-y-1">
                <CardHeader>
                  <div className={`w-12 h-12 ${service.color} rounded-lg flex items-center justify-center mb-4`}>
                    <Icon className="h-6 w-6 text-white" />
                  </div>
                  <CardTitle className="text-xl">{service.title}</CardTitle>
                  <CardDescription className="text-gray-600">
                    {service.description}
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    {service.features.map((feature, featureIndex) => (
                      <div key={featureIndex} className="flex items-center space-x-2 text-sm text-gray-600">
                        <CheckCircle className="h-4 w-4 text-green-500" />
                        <span>{feature}</span>
                      </div>
                    ))}
                  </div>
                  <Button asChild className="w-full">
                    <Link to={service.path} className="flex items-center justify-center space-x-2">
                      <span>Try Now</span>
                      <ArrowRight className="h-4 w-4" />
                    </Link>
                  </Button>
                </CardContent>
              </Card>
            )
          })}
        </div>
      </section>

      {/* CTA Section */}
      <section className="bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl p-8 md:p-12 text-center text-white">
        <div className="space-y-6">
          <h2 className="text-3xl md:text-4xl font-bold">Ready to Get Started?</h2>
          <p className="text-xl opacity-90 max-w-2xl mx-auto">
            Experience the future of healthcare with our AI-powered prediction system. 
            Get instant, accurate health assessments in seconds.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" variant="secondary" className="bg-white text-blue-600 hover:bg-gray-100">
              <Link to="/diabetes" className="flex items-center space-x-2">
                <span>Start Your First Prediction</span>
                <ArrowRight className="h-4 w-4" />
              </Link>
            </Button>
            <Button size="lg" variant="outline" className="border-white text-white hover:bg-white hover:text-blue-600">
              <Link to="/about">Learn About Our Technology</Link>
            </Button>
          </div>
        </div>
      </section>
    </div>
  )
}

export default HomePage

