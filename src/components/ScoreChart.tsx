import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

type Props = {
  data: { time: number; score: number }[];
};

const ScoreChart: React.FC<Props> = ({ data }) => (
  <ResponsiveContainer width="100%" height={200}>
    <LineChart data={data}>
      <XAxis dataKey="time" />
      <YAxis domain={[0, 100]} />
      <Tooltip />
      <Line type="monotone" dataKey="score" stroke="#10b981" strokeWidth={2} dot={false} />
    </LineChart>
  </ResponsiveContainer>
);

export default ScoreChart;