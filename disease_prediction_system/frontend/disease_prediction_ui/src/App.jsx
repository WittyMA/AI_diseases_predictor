import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { Toaster } from '@/components/ui/toaster'
import './App.css'

// Import components
import Header from './components/Header'
import HomePage from './components/HomePage'
import DiabetesPrediction from './components/DiabetesPrediction'
import HeartDiseasePrediction from './components/HeartDiseasePrediction'
import CovidSymptomsPrediction from './components/CovidSymptomsPrediction'
import ChestXrayPrediction from './components/ChestXrayPrediction'
import MultimodalPrediction from './components/MultimodalPrediction'
import AboutPage from './components/AboutPage'

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        <Header />
        <main className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/diabetes" element={<DiabetesPrediction />} />
            <Route path="/heart-disease" element={<HeartDiseasePrediction />} />
            <Route path="/covid-symptoms" element={<CovidSymptomsPrediction />} />
            <Route path="/chest-xray" element={<ChestXrayPrediction />} />
            <Route path="/multimodal" element={<MultimodalPrediction />} />
            <Route path="/about" element={<AboutPage />} />
          </Routes>
        </main>
        <Toaster />
      </div>
    </Router>
  )
}

export default App

