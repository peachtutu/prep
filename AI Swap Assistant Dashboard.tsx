import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { AlertCircle, TrendingUp, Activity, Clock } from 'lucide-react';

const AISwapAssistant = () => {
  const gasData = [
    { time: '00:00', gas: 45 },
    { time: '04:00', gas: 32 },
    { time: '08:00', gas: 68 },
    { time: '12:00', gas: 85 },
    { time: '16:00', gas: 73 },
    { time: '20:00', gas: 52 }
  ];

  return (
    <div className="w-full max-w-4xl p-4 space-y-4">
      {/* Main Dashboard Header */}
      <Card className="bg-white">
        <CardHeader>
          <CardTitle className="text-xl font-bold">AI Swap Assistant</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Gas Price Optimizer */}
            <Card className="p-4">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center">
                  <Clock className="w-5 h-5 mr-2 text-blue-500" />
                  <h3 className="font-semibold">Optimal Swap Timing</h3>
                </div>
                <AlertCircle className="w-5 h-5 text-green-500" />
              </div>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={gasData}>
                    <XAxis dataKey="time" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="gas" stroke="#2563eb" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-2 text-sm text-gray-600">
                Recommended swap time: 04:00 UTC (32 Gwei)
              </div>
            </Card>

            {/* Market Sentiment */}
            <Card className="p-4">
              <div className="flex items-center mb-4">
                <TrendingUp className="w-5 h-5 mr-2 text-blue-500" />
                <h3 className="font-semibold">Market Sentiment</h3>
              </div>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span>Social Sentiment</span>
                  <span className="text-green-500">Bullish</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>On-chain Activity</span>
                  <span className="text-yellow-500">Neutral</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>Price Correlation</span>
                  <span className="text-green-500">Strong</span>
                </div>
              </div>
            </Card>

            {/* Portfolio Impact */}
            <Card className="p-4">
              <div className="flex items-center mb-4">
                <Activity className="w-5 h-5 mr-2 text-blue-500" />
                <h3 className="font-semibold">Portfolio Impact</h3>
              </div>
              <div className="space-y-2">
                <div className="bg-blue-100 p-2 rounded">
                  <div className="text-sm font-medium">Current Position</div>
                  <div className="text-lg">15% of Portfolio</div>
                </div>
                <div className="bg-green-100 p-2 rounded">
                  <div className="text-sm font-medium">Recommended Max</div>
                  <div className="text-lg">25% of Portfolio</div>
                </div>
              </div>
            </Card>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default AISwapAssistant;
